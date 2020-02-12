import os, sys, shutil, time
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.simple_moving_mnist
import tflib.plot
import tflib.objs.gan_inference
import tflib.utils.distance


'''
hyperparameters
'''
# model type
MODE = 'local_epce-z' # local_ep, local_epce-z, ali, alice-z
POS_MODE = 'naive_mean_field' # gsp, naive_mean_field, inverse
ALI_MODE = 'concat_x' # concat_x, concat_z, 3dcnn
OP_DYN_MODE = 'res' # res, res_w
BN_FLAG = False
BN_FLAG_G = BN_FLAG # batch norm in G
BN_FLAG_E = BN_FLAG # batch norm in E
BN_FLAG_D = BN_FLAG # batch norm in D
# model size
DIM_LATENT_G = 128 # global latent variable
DIM_LATENT_L = 32 # local latent variable
DIM_LATENT_T = DIM_LATENT_L # transformation latent variable
DIM = 32 # model size of frame generator
DIM_OP = 256 # model size of the dynamic operator
# data
LEN = 160 # data length
OUTPUT_SHAPE = [1,512] # data shape
OUTPUT_DIM = np.prod(OUTPUT_SHAPE) # data dim
N_C = 12 # number of classes
# optimization
LAMBDA = 0.1 # reconstruction
LR = 1e-4 # learning rate
BATCH_SIZE = 48 # batch size
BETA1 = .5 # adam
BETA2 = .999 # adam
ITERS = 100000 # number of iterations to train
CRITIC_ITERS = 1
# visualization
N_VIS = BATCH_SIZE
assert N_VIS % N_C == 0


'''
logs
'''
filename_script=os.path.basename(os.path.realpath(__file__))
outf=os.path.join("result", os.path.splitext(filename_script)[0])
outf+='.MODE-'
outf+=str(MODE)
outf+='.ALI_MODE-'
outf+=str(ALI_MODE)
outf+='.LEN-'
outf+=str(LEN)
outf+='.'
outf+=str(int(time.time()))
if not os.path.exists(outf):
    os.makedirs(outf)
logfile=os.path.join(outf, 'logfile.txt')
shutil.copy(os.path.realpath(__file__), os.path.join(outf, filename_script))
lib.print_model_settings_to_file(locals().copy(), logfile)

ratio = [1, LEN]
ratio = [1]

ratio = np.asarray(ratio) * 1.0 / (len(ratio))


'''
models
'''
def binarize_labels(y):
    new_y = np.zeros((y.shape[0], N_C))
    for i in range(y.shape[0]):
        new_y[i, y[i]] = 1
    return new_y.astype(np.float32)

def expand_labels(y,l=LEN):
    new_y = tf.tile(tf.expand_dims(y, axis=1), [1, l, 1])
    return tf.reshape(new_y, [-1, N_C])

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ImplicitOperator(z_l, epsilon, name):
    output = tf.concat([z_l, epsilon], axis=1)
    output = lib.ops.linear.Linear(name+'.Input', DIM_LATENT_L+DIM_LATENT_T, DIM_OP, output)
    output = LeakyReLU(output)

    output = lib.ops.linear.Linear(name+'.1', DIM_OP, DIM_OP, output)
    output = LeakyReLU(output)
    
    output = lib.ops.linear.Linear(name+'.Output', DIM_OP, DIM_LATENT_L, output)

    if OP_DYN_MODE == 'res':
        output = output + z_l
    
    elif OP_DYN_MODE == 'res_w':
        output = output + lib.ops.linear.Linear(name+'.ZW', DIM_LATENT_L, DIM_LATENT_L, z_l)

    return output

def ConcatOperator(z_l_0, z_l_1_pre, name):
    output = tf.concat([z_l_0, z_l_1_pre], axis=1)
    output = lib.ops.linear.Linear(name+'.Input', DIM_LATENT_L*2, DIM_OP, output)
    output = LeakyReLU(output)

    output = lib.ops.linear.Linear(name+'.1', DIM_OP, DIM_OP, output)
    output = LeakyReLU(output)

    output = lib.ops.linear.Linear(name+'.Output', DIM_OP, DIM_LATENT_L, output)

    if OP_DYN_MODE == 'res':
        output = z_l_0 + output
    
    elif OP_DYN_MODE == 'res_w':
        output = output + lib.ops.linear.Linear(name+'.ZW', DIM_LATENT_L, DIM_LATENT_L, z_l_0)

    return output

def DynamicGenerator(z_l_0):
    z_list = [z_l_0,]

    epsilon = tf.random_normal([BATCH_SIZE, DIM_LATENT_T])
    for i in xrange(LEN-1):
        z_list.append(ImplicitOperator(z_list[-1], epsilon, 'Generator.Dynamic'))

    return tf.reshape(tf.concat(z_list, axis=1), [BATCH_SIZE, LEN, DIM_LATENT_L])

def DynamicExtractor(z_l_pre):
    if POS_MODE is 'inverse':
        z_list = [z_l_pre[:,LEN - 1,:],]
        for i in xrange(LEN-1):
            z_list.insert(0, ConcatOperator(z_list[0], z_l_pre[:,LEN - i - 2,:], 'Extractor.Dynamic.Backward'))

    elif POS_MODE is 'forward_inverse':
        z_list = [z_l_pre[:,0,:],]
        for i in xrange(LEN-1):
            z_list.append(ConcatOperator(z_list[-1], z_l_pre[:,i + 1,:], 'Extractor.Dynamic.Forward'))

    elif POS_MODE is 'gsp':
        tmp_z_list = [z_l_pre[:,LEN - 1,:],]
        for i in xrange(LEN-1):
            tmp_z_list.insert(0, ConcatOperator(tmp_z_list[0], z_l_pre[:,LEN - i - 2,:], 'Extractor.Dynamic.Backward'))
        z_list = [tmp_z_list[0],]
        for i in xrange(LEN-1):
            z_list.append(ConcatOperator(z_list[-1], tmp_z_list[i + 1], 'Extractor.Dynamic.Forward'))

    elif POS_MODE is 'naive_mean_field':
        return z_l_pre

    else:
        raise('NotImplementedError')

    return tf.reshape(tf.concat(z_list, axis=1), [BATCH_SIZE, LEN, DIM_LATENT_L])




def Generator(z_g, z_l, labels):
    new_output_shape=OUTPUT_SHAPE[-1]/(2**4)
    z_g = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
    z_g = tf.tile(tf.expand_dims(z_g, axis=1), [1, new_output_shape*LEN, 1])
    
    z_l = tf.reshape(z_l, [BATCH_SIZE, new_output_shape*LEN, DIM_LATENT_L])
    labels = expand_labels(labels,l=new_output_shape*LEN)
    labels = tf.reshape(labels, [BATCH_SIZE, LEN*new_output_shape, N_C])
    z = tf.concat([z_g, z_l], axis=-1)

    z = tf.reshape(z, [BATCH_SIZE*LEN*new_output_shape, DIM_LATENT_G+DIM_LATENT_L])

    output = lib.ops.linear.Linear('Generator.Input', DIM_LATENT_G+DIM_LATENT_L, 8*DIM, z)
    if BN_FLAG_G:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [BATCH_SIZE, 8*DIM, new_output_shape*LEN])

    output = lib.ops.deconv1d.Deconv1D('Generator.2', 8*DIM, 4*DIM, 5, output)
    if BN_FLAG_G:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv1d.Deconv1D('Generator.3', 4*DIM, 2*DIM, 5, output)
    if BN_FLAG_G:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv1d.Deconv1D('Generator.4', 2*DIM, DIM, 5, output)
    if BN_FLAG_G:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0,2], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv1d.Deconv1D('Generator.5', DIM, 1, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [BATCH_SIZE, LEN, OUTPUT_DIM])

def Extractor(inputs, labels):
    new_output_shape=OUTPUT_SHAPE[-1]/(2**4)
    output = tf.reshape(inputs, [BATCH_SIZE,1,OUTPUT_DIM*LEN] )
    labels = expand_labels(labels,l=new_output_shape*LEN)
    labels = tf.reshape(labels, [BATCH_SIZE*LEN*new_output_shape,N_C])
    output = lib.ops.conv1d.Conv1D('Extractor.1', 1, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv1d.Conv1D('Extractor.2', DIM, 2*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.BN2', [0,2], output)
    output = LeakyReLU(output)

    output = lib.ops.conv1d.Conv1D('Extractor.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.BN3', [0,2], output)
    output = LeakyReLU(output)

    output = lib.ops.conv1d.Conv1D('Extractor.4', 4*DIM, 8*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.BN4', [0,2], output)
    output = LeakyReLU(output)
    new_output_shape=OUTPUT_SHAPE[-1]/(2**4)
    output = tf.reshape(output, [BATCH_SIZE*LEN*new_output_shape, 8*DIM])

    # output = tf.concat([output, labels], axis=1)

    output = lib.ops.linear.Linear('Extractor.Output', 8*DIM, DIM_LATENT_L, output)

    return tf.reshape(output, [BATCH_SIZE, LEN*new_output_shape, DIM_LATENT_L])

def G_Extractor(inputs, labels):
    output = tf.reshape(inputs, [BATCH_SIZE, LEN,]+ [OUTPUT_SHAPE[-1]])

    output = lib.ops.conv1d.Conv1D('Extractor.G.1', LEN, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv1d.Conv1D('Extractor.G.2', DIM, 2*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.G.BN2', [0,2], output)
    output = LeakyReLU(output)

    output = lib.ops.conv1d.Conv1D('Extractor.G.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.G.BN3', [0,2], output)
    output = LeakyReLU(output)

    output = lib.ops.conv1d.Conv1D('Extractor.G.4', 4*DIM, 8*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.G.BN4', [0,2], output)
    output = LeakyReLU(output)
    new_output_shape=OUTPUT_SHAPE[-1]/(2**4)
    output = tf.reshape(output, [BATCH_SIZE, new_output_shape*8*DIM])
    # output = tf.concat([output, labels], axis=1)
    output = lib.ops.linear.Linear('Extractor.G.Output', new_output_shape*8*DIM, DIM_LATENT_G, output)

    return tf.reshape(output, [BATCH_SIZE, DIM_LATENT_G])

def g_Classifier(z_g):
    new_output_shape=OUTPUT_SHAPE[-1]/(2**4)
    output = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
    output = LeakyReLU(output)
    output = tf.layers.dropout(output, rate=.2)
    output=lib.ops.linear.Linear('Classifier.G.Output', DIM_LATENT_G, N_C, output)
    return tf.reshape(output, [BATCH_SIZE, N_C])

def l_Classifier(z_l):
    new_output_shape=OUTPUT_SHAPE[-1]/(2**4)
    output=tf.reshape(z_l, [BATCH_SIZE, LEN*DIM_LATENT_L, new_output_shape])
    output = lib.ops.conv1d.Conv1D('Classifier.L.1', LEN*DIM_LATENT_L, DIM, 5, output, stride=2)
    output = LeakyReLU(output)
    output = tf.reshape(output, [BATCH_SIZE, new_output_shape/2*DIM])
    output = lib.ops.linear.Linear('Classifier.L.Output', new_output_shape/2*DIM, N_C, output)
    return output




if MODE in ['local_ep', 'local_epce-z']:
    def Discriminator(x, z_g, z_l, labels):
        output = tf.reshape(x, [BATCH_SIZE*LEN,] + OUTPUT_SHAPE)
        labels = expand_labels(labels)
        labels = tf.reshape(labels, [BATCH_SIZE, LEN, N_C])
        z_g = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
        z_g = tf.tile(tf.expand_dims(z_g, axis=1), [1, LEN, 1])
        z_l = tf.reshape(z_l, [BATCH_SIZE, LEN, DIM_LATENT_L])
        z = tf.concat([z_g, z_l, labels], axis=-1)
        z = tf.reshape(z, [BATCH_SIZE*LEN, DIM_LATENT_G+DIM_LATENT_L+N_C])

        output = lib.ops.conv1d.Conv1D('Discriminator.1', 1, DIM, 5,output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv1d.Conv1D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
        if BN_FLAG_D:
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2], output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv1d.Conv1D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
        if BN_FLAG_D:
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2], output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv1d.Conv1D('Discriminator.4', 4*DIM, 8*DIM, 5, output, stride=2)
        if BN_FLAG_D:
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,2], output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)
        new_output_shape=OUTPUT_SHAPE[-1]/(2**4)
        output = tf.reshape(output, [BATCH_SIZE*LEN, new_output_shape*8*DIM])

        z_output = lib.ops.linear.Linear('Discriminator.z1', DIM_LATENT_G+DIM_LATENT_L+N_C, 512, z)
        z_output = LeakyReLU(z_output)
        z_output = tf.layers.dropout(z_output, rate=.2)

        labels = tf.reshape(labels, [BATCH_SIZE*LEN, N_C])
        output = tf.concat([output, z_output, labels], 1)
        output = lib.ops.linear.Linear('Discriminator.zx1', new_output_shape*8*DIM+512+N_C, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

        return tf.reshape(output, [BATCH_SIZE*LEN,])

    def DynamicDiscrminator(z1, z2):
        z1 = tf.reshape(z1, [BATCH_SIZE, DIM_LATENT_L])
        z2 = tf.reshape(z2, [BATCH_SIZE, DIM_LATENT_L])
        output = tf.concat([z1, z2], axis=1)
        output = lib.ops.linear.Linear('Discriminator.Dynamic.Input', DIM_LATENT_L*2, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Dynamic.2', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Dynamic.3', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Dynamic.Output', 512, 1, output)

        return tf.reshape(output, [BATCH_SIZE,])

    def ZGDiscrminator(z_g):
        output = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
        output = lib.ops.linear.Linear('Discriminator.ZG.Input', DIM_LATENT_G, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.ZG.2', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.ZG.3', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.ZG.Output', 512, 1, output)

        return tf.reshape(output, [BATCH_SIZE,])

elif MODE in ['ali', 'alice-z']:
    if ALI_MODE is '3dcnn':
        import tflib.ops.conv3d
        def Discriminator(x, z_g, z_l, labels):
            output = tf.reshape(x, [-1, LEN] + OUTPUT_SHAPE)
            output = tf.transpose(output, [0, 1, 3, 4, 2]) #NLHWC

            z_l = tf.reshape(z_l, [BATCH_SIZE, LEN*DIM_LATENT_L])
            z_g = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
            labels = tf.reshape(labels, [BATCH_SIZE, N_C])
            z = tf.concat([z_g, z_l, labels], axis=-1)

            output = lib.ops.conv3d.Conv3D('Discriminator.1', 4, 1, DIM, 4, output, stride=2, stride_len=2)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            if LEN == 4:
                output = lib.ops.conv3d.Conv3D('Discriminator.2', 4, DIM, 2*DIM, 4, output, stride=2, stride_len=1)
            elif LEN == 16:
                output = lib.ops.conv3d.Conv3D('Discriminator.2', 4, DIM, 2*DIM, 4, output, stride=2, stride_len=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,1,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv3d.Conv3D('Discriminator.3', 4, 2*DIM, 4*DIM, 4, output, stride=2, stride_len=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,1,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            if LEN == 4:
                output = lib.ops.conv3d.Conv3D('Discriminator.4', 4, 4*DIM, 8*DIM, 4, output, stride=2, stride_len=1)
            elif LEN == 16:
                output = lib.ops.conv3d.Conv3D('Discriminator.4', 4, 4*DIM, 8*DIM, 4, output, stride=2, stride_len=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,1,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = tf.reshape(output, [BATCH_SIZE, 4*4*8*DIM])

            z_output = lib.ops.linear.Linear('Discriminator.z1', DIM_LATENT_G+DIM_LATENT_L*LEN+N_C, 512, z)
            z_output = LeakyReLU(z_output)
            z_output = tf.layers.dropout(z_output, rate=.2)

            output = tf.concat([output, z_output], 1)
            output = lib.ops.linear.Linear('Discriminator.zx1', 4*4*8*DIM+512, 512, output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

            return tf.reshape(output, [BATCH_SIZE,])

    elif ALI_MODE is 'concat_x':
        def Discriminator(x, z_g, z_l, labels):
            output = tf.reshape(x, [BATCH_SIZE, LEN, 64, 64])
            
            z_l = tf.reshape(z_l, [BATCH_SIZE, LEN*DIM_LATENT_L])
            z_g = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
            labels = tf.reshape(labels, [BATCH_SIZE, N_C])
            z = tf.concat([z_g, z_l, labels], axis=-1)

            output = lib.ops.conv2d.Conv2D('Discriminator.1', LEN, DIM, 5, output, stride=2)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM, 8*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = tf.reshape(output, [BATCH_SIZE, 4*4*8*DIM])

            z_output = lib.ops.linear.Linear('Discriminator.z1', DIM_LATENT_G+DIM_LATENT_L*LEN+N_C, 512, z)
            z_output = LeakyReLU(z_output)
            z_output = tf.layers.dropout(z_output, rate=.2)

            output = tf.concat([output, z_output], 1)
            output = lib.ops.linear.Linear('Discriminator.zx1', 4*4*8*DIM+512, 512, output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

            return tf.reshape(output, [BATCH_SIZE,])

    elif ALI_MODE is 'concat_z':
        def Discriminator(x, z_g, z_l, labels):
            output = tf.reshape(x, [BATCH_SIZE*LEN, -1, 64, 64])
            
            z_l = tf.reshape(z_l, [BATCH_SIZE, LEN*DIM_LATENT_L])
            z_g = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
            labels = tf.reshape(labels, [BATCH_SIZE, N_C])
            z = tf.concat([z_g, z_l, labels], axis=-1)

            output = lib.ops.conv2d.Conv2D('Discriminator.1', 1, DIM, 5, output, stride=2)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM, 8*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.5', 8*DIM, DIM_LATENT_G, 4, output, stride=1, padding='VALID')

            output = tf.reshape(output, [BATCH_SIZE, LEN*DIM_LATENT_G])

            z_output = lib.ops.linear.Linear('Discriminator.z1', DIM_LATENT_G+DIM_LATENT_L*LEN+N_C, 512, z)
            z_output = LeakyReLU(z_output)
            z_output = tf.layers.dropout(z_output, rate=.2)

            output = tf.concat([output, z_output, labels], 1)
            output = lib.ops.linear.Linear('Discriminator.zx1', LEN*DIM_LATENT_G+512+N_C, 512, output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

            return tf.reshape(output, [BATCH_SIZE,])
    
    else:
        raise('NotImplementedError')

else:
    raise('NotImplementedError')


'''
losses
'''
PI = tf.constant(np.asarray([1./N_C,]*N_C, dtype=np.float32))
prior_y = tf.distributions.Categorical(probs=PI)

real_x_unit = tf.placeholder(tf.float32, shape=[BATCH_SIZE, LEN, OUTPUT_DIM])
real_x = real_x_unit/15000.0
real_y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_C])
q_z_l_pre = Extractor(real_x, real_y)
q_z_g = G_Extractor(real_x, real_y)
q_z_l = q_z_l_pre#DynamicExtractor(q_z_l_pre)
rec_x = Generator(q_z_g, q_z_l, real_y)

# p_z_l_0 = tf.random_normal([BATCH_SIZE, DIM_LATENT_L])
p_z_l = q_z_l#DynamicGenerator(p_z_l_0)
p_z_g = tf.random_normal([BATCH_SIZE, DIM_LATENT_G])
p_y = tf.one_hot(indices=prior_y.sample(BATCH_SIZE), depth=N_C)
fake_x = Generator(p_z_g, p_z_l, p_y)





if MODE in ['local_ep', 'local_epce-z']:
    disc_fake, disc_real = [],[]
    # for i in xrange(LEN-1):
    #     disc_fake.append(DynamicDiscrminator(p_z_l[:,i,:], p_z_l[:,i+1,:]))
    #     disc_real.append(DynamicDiscrminator(q_z_l[:,i,:], q_z_l[:,i+1,:]))
    disc_fake.append(ZGDiscrminator(p_z_g))
    disc_real.append(ZGDiscrminator(q_z_g))
    # disc_fake.append(Discriminator(fake_x, p_z_g, p_z_l, p_y))
    # disc_real.append(Discriminator(real_x, q_z_g, q_z_l, real_y))

elif MODE in ['ali', 'alice-z']:
    disc_real = Discriminator(real_x, q_z_g, q_z_l, real_y)
    disc_fake = Discriminator(fake_x, p_z_g, p_z_l, p_y)

gen_params = lib.params_with_name('Generator')
ext_params = lib.params_with_name('Extractor')
disc_params = lib.params_with_name('Discriminator')
local_classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=l_Classifier(q_z_l),
    labels=real_y,
    name = 'lc'
))

global_classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=g_Classifier(q_z_g),
    labels=real_y,
    name='gc'
))
print g_Classifier(q_z_g),real_y
classg_params = lib.params_with_name('Classifier.G')
classl_params = lib.params_with_name('Classifier.L')
if MODE == 'local_ep':
    rec_penalty = None
    gen_cost, disc_cost, _, _, gen_train_op, disc_train_op = \
    lib.objs.gan_inference.weighted_local_epce(disc_fake, 
        disc_real, 
        local_classifier_loss,
        global_classifier_loss,
        ratio,
        gen_params+ext_params, 
        disc_params, 
        lr=LR, 
        beta1=BETA1, 
        rec_penalty=rec_penalty)

elif MODE == 'local_epce-z':
    rec_penalty = LAMBDA*lib.utils.distance.distance(real_x, rec_x, 'l2')
    gen_cost, disc_cost, _, _, gen_train_op, disc_train_op, cl_train_op, cg_train_op = \
    lib.objs.gan_inference.weighted_local_epce(disc_fake, 
        disc_real,
        local_classifier_loss,
        global_classifier_loss, 
        classg_params,
        classl_params,
        ratio, 
        gen_params+ext_params, 
        disc_params, 
        lr=LR, 
        beta1=BETA1, 
        rec_penalty=rec_penalty)

elif MODE == 'ali':
    rec_penalty = None
    gen_cost, disc_cost, gen_train_op, disc_train_op = lib.objs.gan_inference.ali(disc_fake, disc_real, gen_params+ext_params, disc_params, lr=LR, beta1=BETA1, beta2=BETA2)

elif MODE == 'alice-z':
    rec_penalty = LAMBDA*lib.utils.distance.distance(real_x, rec_x, 'l2')
    gen_cost, disc_cost, gen_train_op, disc_train_op = lib.objs.gan_inference.alice(disc_fake, disc_real, rec_penalty, gen_params+ext_params, disc_params, lr=LR, beta1=BETA1)

# Dataset iterator
train_gen, dev_gen = lib.simple_phone.load_audio(OUTPUT_DIM,LEN,BATCH_SIZE)
def inf_train_gen():
    while True:
        for audios, labels in train_gen():
            yield audios, binarize_labels(labels)

# For visualization
def wav(x, iteration, num, name):
    x = x.reshape((num, -1))
    import os
    dir=os.path.join(outf, name+'_'+str(iteration)+'/')
    if not os.path.isdir(dir):
        os.mkdir(dir)

    lib.save_wavs.save_wavs(x, dir, size=None)

# For generation
fixed_data, fixed_y = dev_gen().next()
print fixed_y 
fixed_y = binarize_labels(fixed_y)
pre_fixed_noise = tf.constant(np.random.normal(size=(N_VIS, DIM_LATENT_L)).astype('float32'))
fixed_y = tf.constant(np.tile(np.eye(N_C, dtype=int), (N_VIS/N_C, 1)).astype(np.float32))
fixed_noise_g = tf.constant(np.random.normal(size=(N_VIS, DIM_LATENT_G)).astype('float32'))
fixed_noise_l = q_z_l#DynamicGenerator(pre_fixed_noise)
fixed_noise_samples = Generator(fixed_noise_g, fixed_noise_l, fixed_y)
def generate_video(iteration, data):

    samples = session.run(fixed_noise_samples, feed_dict={real_x_unit: fixed_data, real_y:fixed_y})
    samples = samples*15000.0
    samples = samples[:int(N_VIS/4)]
    wav(samples, iteration, N_VIS/4, 'samples')
    # wav(data, iteration, BATCH_SIZE/4, 'train_data')

# For reconstruction
fixed_data, fixed_y = dev_gen().next()
fixed_y = binarize_labels(fixed_y)
def reconstruct_video(iteration):
    rec_samples = session.run(rec_x, feed_dict={real_x_unit: fixed_data, real_y:fixed_y})
    rec_samples = rec_samples*15000.0
    rec_samples = rec_samples.reshape((-1, LEN, OUTPUT_DIM))
    tmp_list = []
    for i in xrange(BATCH_SIZE/4):
        tmp_list.append(fixed_data[i])
        tmp_list.append(rec_samples[i])
    rec_samples = np.vstack(tmp_list)
    wav(rec_samples, iteration, BATCH_SIZE/2, 'reconstruction')

# disentangle
fixed_data, fixed_y = dev_gen().next()
fixed_y = binarize_labels(fixed_y)
print fixed_y
dis_y = tf.constant(binarize_labels(np.ones(BATCH_SIZE, dtype=int)))
dis_g = tf.constant(np.tile(np.random.normal(size=(1, DIM_LATENT_G)).astype('float32'), [BATCH_SIZE, 1]))
dis_x = Generator(dis_g, q_z_l, dis_y)
def disentangle(iteration):
    samples = session.run(dis_x, feed_dict={real_x_unit: fixed_data, real_y:fixed_y})
    samples = samples*15000.0
    tmp_list = []
    for i in xrange(BATCH_SIZE/4):
        tmp_list.append(fixed_data[i])
        tmp_list.append(samples[i])
    samples = np.vstack(tmp_list)
    wav(samples, iteration, BATCH_SIZE/2, 'disentangle')


'''
Train loop
'''
saver = tf.train.Saver()
with tf.Session() as session:

    session.run(tf.global_variables_initializer())
    gen = inf_train_gen()

    total_num = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print '\nTotol number of parameters', total_num
    with open(logfile,'a') as f:
        f.write('\nTotol number of parameters' + str(total_num) + '\n')

    gen_num = tf.reduce_sum([tf.reduce_prod(tf.shape(t)) for t in gen_params])
    ext_num = tf.reduce_sum([tf.reduce_prod(tf.shape(t)) for t in ext_params])
    disc_num = tf.reduce_sum([tf.reduce_prod(tf.shape(t)) for t in disc_params])

    print '\nNumber of parameters in each player', session.run([gen_num, ext_num, disc_num, gen_num+ext_num+disc_num]), '\n'
    with open(logfile,'a') as f:
        f.write('\nNumber of parameters in each player' + str(session.run([gen_num, ext_num, disc_num, gen_num+ext_num+disc_num])) + '\n')

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _data, _labels = gen.next()
            print _labels
            if rec_penalty is None:
                _gen_cost, _ = session.run([gen_cost, gen_train_op],
                feed_dict={real_x_unit: _data, real_y:_labels})
            else:
                _gen_cost, _rec_cost, _ = session.run([gen_cost, rec_penalty, gen_train_op],
                feed_dict={real_x_unit: _data, real_y:_labels})
            
        for i in xrange(CRITIC_ITERS):
            _data, _labels = gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_x_unit: _data, real_y:_labels}
            )
            _cg_cost, _ = session.run(
                [global_classifier_loss, cg_train_op],
                feed_dict={real_x_unit: _data, real_y:_labels}
            )
            _cl_cost, _ = session.run(
                [local_classifier_loss, local_train_op],
                feed_dict={real_x_unit: _data, real_y:_labels}
            )
        if iteration > 0:
            lib.plot.plot('gc', _gen_cost)
            lib.plot.plot('cg', _cg_cost)
            lib.plot.plot('cl', _cl_cost)
            if rec_penalty is not None:
                lib.plot.plot('rc', _rec_cost)
        lib.plot.plot('dc', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Write logs
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush(outf, logfile)
        lib.plot.tick()

        # Generation and reconstruction
        if iteration % 1000 == 999:
            generate_video(iteration, _data)
            reconstruct_video(iteration)
            disentangle(iteration)

        # Save model
        if iteration == ITERS - 1:
            save_path = saver.save(session, os.path.join(outf, '{}_model_{}.ckpt'.format(iteration, MODE)))
