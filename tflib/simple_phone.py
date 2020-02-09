import numpy as np

import os
import urllib
import gzip
import cPickle as pickle


def GetRandomTrajectory(step_length, seq_length, batch_size, image_size,digit_size):
    canvas_size = image_size - digit_size
    
    # Initial position uniform random inside the box.
    y = np.random.rand(batch_size)
    x = np.random.rand(batch_size)

    # Choose a random velocity.
    theta = np.random.rand(batch_size) * 2 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros((seq_length, batch_size))
    start_x = np.zeros((seq_length, batch_size))
    for i in xrange(seq_length):
        # Take a step along velocity.
        y += v_y * step_length
        x += v_x * step_length

        # Bounce off edges.
        for j in xrange(batch_size):
            if x[j] <= 0:
                x[j] = 0
                v_x[j] = -v_x[j]
            if x[j] >= 1.0:
                x[j] = 1.0
                v_x[j] = -v_x[j]
            if y[j] <= 0:
                y[j] = 0
                v_y[j] = -v_y[j]
            if y[j] >= 1.0:
                y[j] = 1.0
                v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    return start_y, start_x

def Overlap(a, b):
    return np.maximum(a, b)
    #return b

def phone_generator_audio(data_all, clip_length, seq_length, batch_size):
    audios_pre, labels = data_all
    # images = images.reshape([-1, 28, 28])
    # image_size = 64
    # num_digits = 1
    # step_length = 0.1
    # digit_size = 28
    
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(audios)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        audios=[]
        for a in audios_pre:
            start=np.random.randint(0,len(a))
            audios.append(np.concatenate((a[start:],a[:start])))

        
        audios=[np.squeeze(a) for a in audios]

        # start_y, start_x = GetRandomTrajectory(step_length = step_length, seq_length = seq_length, batch_size = images.shape[0]*num_digits, image_size = image_size, digit_size = digit_size)

        data = []
        data_label = []
        for i in xrange(int(len(audios[0])/(seq_length*clip_length)*2)):
            data_len=len(data)
            for a,l in zip(audios,labels):
                if i*seq_length*clip_length>=len(a):
                    continue
                new=a[i*seq_length*clip_length   :  min(len(a),(i+1)*seq_length*clip_length)  ]
                if len(new)<seq_length*clip_length:
                    new=np.pad(new,(0,seq_length*clip_length-len(new)),'constant')
                data.append( new )
                data_label.append(l)
            if len(data)==data_len:
                break
            
        data=np.stack(data)
        data = data.reshape(-1, seq_length, clip_length)
        data_label=np.array(data_label)
        for ind in xrange(data.shape[0]/ batch_size):
             yield data[ind*batch_size:(ind+1)*batch_size], data_label[ind*batch_size:(ind+1)*batch_size]

    return get_epoch

def load_audio(clip_length,seq_length, batch_size, cla=None):
    filepath = '/tmp/phone.pkl.gz'
    # url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        # urllib.urlretrieve(url, filepath)
    with gzip.open('/tmp/phone.pkl.gz', 'rb') as f:
        x, y = pickle.load(f)
    train_all_x = [i[:int(len(i)/3*2)] for i in x]
    train_all_y = y
    test_x = [i[int(len(i)/3*2):] for i in x]
    test_y = y
    test_data=(test_x,test_y)



    return (phone_generator_audio((train_all_x, train_all_y), clip_length,seq_length, batch_size), phone_generator_audio(test_data, clip_length,seq_length, batch_size))

def moving_mnist_generator_image(image, seq_length, batch_size):
    assert batch_size % seq_length == 0
    video_gen = moving_mnist_generator_video(image, seq_length, batch_size/seq_length)
    data = []
    label = []
    for v, y in video_gen():
        data.append(v.reshape([batch_size, 64*64]))
        label.append(np.tile(y.reshape(-1, 1), [1, seq_length]).reshape(-1))
    data = np.vstack(data)
    label = np.concatenate(label, axis=0)
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(rng_state)
        np.random.shuffle(label)

        for i in xrange(len(data) / batch_size):
            yield data[i*batch_size:(i+1)*batch_size], label[i*batch_size:(i+1)*batch_size]
    return get_epoch

def load_image(seq_length, batch_size, cla=None):
    filepath = '/tmp/phone.pkl.gz'
    # url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        # urllib.urlretrieve(url, filepath)
    with gzip.open('/tmp/phone.pkl.gz', 'rb') as f:
        x, y = pickle.load(f)
    train_all_x = [i[:int(len(i)/3*2)] for i in x]
    train_all_y = y
    test_x = [i[int(len(i)/3*2):] for i in x]
    test_y = y

    if cla is not None:
        train_all_x = train_all_x[train_all_y == cla]
        train_all_y = train_all_y[train_all_y == cla]
        test_x, test_y = test_data
        test_x = test_x[test_y == cla]
        test_y = test_y[test_y == cla]
        test_data = (test_x, test_y)

    return (moving_mnist_generator_image((train_all_x, train_all_y), seq_length, batch_size), moving_mnist_generator_image(test_data, seq_length, batch_size))