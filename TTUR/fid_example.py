#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from . import fid
from scipy.misc import imread
import tensorflow as tf

def calc_fid(image_path="data2/", stats_path="fid_stats.npz", mu_real=None, sigma_real=None):
    # Paths
    inception_path = fid.check_or_download_inception(None) # download inception network

    # loads all generated images into memory (this might require a lot of RAM!)
    image_list = glob.glob(os.path.join(image_path, '*.jpg'))
    print("num images", len(image_list))
    images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

    if (mu_real is None) or (sigma_real is None):
        # load precalculated training set statistics
        f = np.load(stats_path)
        mu_real, sigma_real = f['mu'][:], f['sigma'][:]
        f.close()

    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)

    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    print("FID: %s" % fid_value)
    return fid_value

if __name__ == "__main__":
    calc_fid()
