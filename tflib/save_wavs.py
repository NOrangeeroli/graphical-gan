"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave
import imageio


def large_image(X, size=None):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    
    if size == None:
        rows = int(np.sqrt(n_samples))
        while n_samples % rows != 0:
            rows -= 1

        nh, nw = rows, n_samples/rows
    else:
        nh, nw = size
        assert(nh * nw == n_samples)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    return img.astype('uint8')

def save_wavs(x, save_path, size=None):
    import wave
    for i, w in enumerate(x):
        f = wave.open(save_path+str(i)+'.wav', "wb")
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(22050)
        w=np.array([int(i) for i in w]).astype(np.short)
        f.writeframes(w.tostring())
        f.close()

def save_images(X, save_path, size=None):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    
    if size == None:
        rows = int(np.sqrt(n_samples))
        while n_samples % rows != 0:
            rows -= 1

        nh, nw = rows, n_samples/rows
    else:
        nh, nw = size
        assert(nh * nw == n_samples)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    imsave(save_path, img)