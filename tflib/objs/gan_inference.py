import tensorflow as tf
import tflib as lib

def wali(disc_fake, disc_real, gen_params, disc_params, lr=5e-5):
    gen_cost = -tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=lr
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=lr
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

    return gen_cost, disc_cost, clip_disc_weights, gen_train_op, disc_train_op, clip_ops

def wali_gp(disc_fake, disc_real, gradient_penalty, gen_params, disc_params, lr=1e-4):
    gen_cost = -tf.reduce_mean(disc_fake) + tf.reduce_mean(disc_real)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    disc_cost += gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)    
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    return gen_cost, disc_cost, gen_train_op, disc_train_op

def ali(disc_fake, disc_real, gen_params, disc_params, lr=2e-4, beta1=0.5, beta2=0.999, s_f = None):
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.ones_like(disc_fake)
    ))
    gen_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real, 
        labels=tf.zeros_like(disc_real)
    ))

    disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real, 
        labels=tf.ones_like(disc_real)
    ))
    if s_f is not None:
        gen_cost += s_f

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1,
        beta2=beta2
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1,
        beta2=beta2
    ).minimize(disc_cost, var_list=disc_params)

    return gen_cost, disc_cost, gen_train_op, disc_train_op

def local_ep(disc_fake_list, disc_real_list, gen_params, disc_params, lr=2e-4, beta1=0.5, beta2=.999, s_f=None):
    gen_cost = 0
    disc_cost = 0
    for disc_fake, disc_real in zip(disc_fake_list, disc_real_list):
        gen_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.ones_like(disc_fake)
        ))
        gen_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, 
            labels=tf.zeros_like(disc_real)
        ))

        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.zeros_like(disc_fake)
        ))
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, 
            labels=tf.ones_like(disc_real)
        ))
    if s_f is not None:
        gen_cost += s_f

    gen_cost /= len(disc_fake_list)
    disc_cost /= len(disc_fake_list)

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1,
        beta2=beta2
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1,
        beta2=beta2
    ).minimize(disc_cost, var_list=disc_params)

    return gen_cost, disc_cost, gen_train_op, disc_train_op

def local_epce(disc_fake_list, disc_real_list, rec_penalty, gen_params, disc_params, lr=2e-4, beta1=0.5, s_f = None):
    gen_cost = 0
    disc_cost = 0
    for disc_fake, disc_real in zip(disc_fake_list, disc_real_list):
        gen_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.ones_like(disc_fake)
        ))
        gen_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, 
            labels=tf.zeros_like(disc_real)
        ))

        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.zeros_like(disc_fake)
        ))
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, 
            labels=tf.ones_like(disc_real)
        ))
    if s_f is not None:
        gen_cost += s_f

    gen_cost /= len(disc_fake_list)
    disc_cost /= len(disc_fake_list)

    gen_cost += rec_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(disc_cost, var_list=disc_params)

    return gen_cost, disc_cost, gen_train_op, disc_train_op

def alice(disc_fake, disc_real, rec_penalty, gen_params, disc_params, lr=2e-4, beta1=0.5, s_f = None):
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.ones_like(disc_fake)
    ))
    gen_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real, 
        labels=tf.zeros_like(disc_real)
    ))
    if s_f is not None:
        gen_cost += s_f
    gen_cost += rec_penalty

    disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real, 
        labels=tf.ones_like(disc_real)
    ))

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(disc_cost, var_list=disc_params)

    return gen_cost, disc_cost, gen_train_op, disc_train_op

def vegan(disc_fake, disc_real, rec_penalty, gen_params, disc_params, lamb, lr=2e-4, beta1=.5, s_f = None):
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.ones_like(disc_fake)
    ))
    if s_f is not None:
        gen_cost += s_f
    gen_cost *= lamb
    gen_cost += rec_penalty

    disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real, 
        labels=tf.ones_like(disc_real)
    ))
    disc_cost *= (lamb/2)

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(disc_cost, var_list=disc_params)

    return gen_cost, disc_cost, gen_train_op, disc_train_op

def vegan_wgan_gp(disc_fake, disc_real, rec_penalty, gradient_penalty, gen_params, disc_params, lamb, lr=2e-4, beta1=.5):
    
    gen_cost = -tf.reduce_mean(disc_fake) + tf.reduce_mean(disc_real)
    gen_cost *= lamb
    gen_cost += rec_penalty

    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    disc_cost *= lamb
    disc_cost += gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(disc_cost, var_list=disc_params)

    return gen_cost, disc_cost, gen_train_op, disc_train_op

def local_ep_dynamic(disc_fake_zz, disc_real_zz, disc_fake_xz, disc_real_xz, gen_params, disc_params, lr=2e-4, beta1=0.5, beta2=.999, rec_penalty=None):
    gen_cost = 0
    disc_cost = 0
    for disc_fake, disc_real in zip(disc_fake_zz, disc_real_zz):
        gen_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.ones_like(disc_fake)
        ))
        gen_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, 
            labels=tf.zeros_like(disc_real)
        ))

        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.zeros_like(disc_fake)
        ))
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, 
            labels=tf.ones_like(disc_real)
        ))

    if len(disc_fake_zz) > 0:
        gen_cost /= (len(disc_fake_zz)+1)
        disc_cost /= (len(disc_fake_zz)+1)
    
    gen_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake_xz, 
        labels=tf.ones_like(disc_fake_xz)
    ))
    gen_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real_xz, 
        labels=tf.zeros_like(disc_real_xz)
    ))

    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake_xz, 
        labels=tf.zeros_like(disc_fake_xz)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real_xz, 
        labels=tf.ones_like(disc_real_xz)
        ))

    if rec_penalty is not None:
        gen_cost += rec_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1,
        beta2=beta2
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1,
        beta2=beta2
    ).minimize(disc_cost, var_list=disc_params)

    return gen_cost, disc_cost, gen_train_op, disc_train_op


def weighted_local_epce(disc_fake_list, 
    disc_real_list,
    local_classifier_loss,
    global_classifier_loss,
    cl_params,
    cg_params,
    ratio_list, 
    gen_params, 
    disc_params,

    lr=2e-4, beta1=0.5, rec_penalty = None):
    gen_cost = 0
    disc_cost = 0

    #assert len(disc_fake_list) == ratio_list.shape[0]
    gen_debug_list, disc_debug_list = [],[]
    for disc_fake, disc_real, ratio in zip(disc_fake_list, disc_real_list, ratio_list):
        gen_cost += ratio * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.ones_like(disc_fake)
        ))
        gen_cost += ratio * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, 
            labels=tf.zeros_like(disc_real)
        ))
        gen_debug_list.append(ratio * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.ones_like(disc_fake)))+ 
            ratio* tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, 
            labels=tf.zeros_like(disc_real)))
        )

        disc_cost += ratio * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.zeros_like(disc_fake)
        ))
        disc_cost += ratio * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, 
            labels=tf.ones_like(disc_real)
        ))
        disc_debug_list.append(ratio * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.zeros_like(disc_fake)))+ 
            ratio * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, 
            labels=tf.ones_like(disc_real)))
        )
    gen_cost*=0
    from math import sqrt

    if rec_penalty is not None:
        gen_cost += rec_penalty#10.0* tf.math.log(rec_penalty)
    gen_cost *= (tf.nn.relu(global_classifier_loss[1]-0.4)  +1.0)
        #gen_cost += tf.math.abs((local_classifier_loss-2.484906649788))
    #gen_cost += tf.nn.relu(1.0/(local_classifier_loss)-1/2.48)+ tf.nn.relu(tf.math.exp(local_classifier_loss)-tf.math.exp(2.48))
    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = None
    '''tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(disc_cost, var_list=disc_params)'''
    cl_train_op = None
    '''tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(local_classifier_loss, var_list=cl_params)'''
    cg_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(global_classifier_loss[0], var_list=cg_params)

    return gen_cost, disc_cost , gen_debug_list, disc_debug_list, gen_train_op, disc_train_op, cl_train_op, cg_train_op



