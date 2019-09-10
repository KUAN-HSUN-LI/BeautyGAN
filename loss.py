from util import *
import tensorflow as tf

import numpy as np

def cycle_consistency_loss(I_src, I_src_rec, I_ref, I_ref_rec):
    dist1 = tf.nn.l2_loss(I_src - I_src_rec)
    dist2 = tf.nn.l2_loss(I_ref - I_ref_rec)
    return tf.add(dist1, dist2) / 256 / 256

def generation_loss(I_src, I_src_B, I_ref, I_ref_A):
    LDA_1 = tf.reduce_mean(tf.squared_difference(I_src, 1))
    LDA_2 = tf.reduce_mean(tf.square(I_ref_A))
    LDB_1 = tf.reduce_mean(tf.squared_difference(I_src_B, 1))
    LDB_2 = tf.reduce_mean(tf.square(I_ref))
    LDA = LDA_1 + LDA_2
    LDB = LDB_1 + LDB_2
    return LDA, LDB

def discrimination_loss(I_src, I_src_B, I_ref, I_ref_A):
    LDA_1 = tf.reduce_mean(tf.square(I_src))
    LDA_2 = tf.reduce_mean(tf.squared_difference(I_ref_A, 1))
    LDB_1 = tf.reduce_mean(tf.square(I_src_B))
    LDB_2 = tf.reduce_mean(tf.squared_difference(I_ref, 1))
    LDA = LDA_1 + LDA_2
    LDB = LDB_1 + LDB_2
    return LDA, LDB

def styleloss(layers):
	gen_f, style_f = tf.split(layers[0], 2, 0)
	size = tf.size(gen_f)
	style_loss = tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

	gen_f, style_f = tf.split(layers[1], 2, 0)
	size = tf.size(gen_f)
	style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

	gen_f, style_f = tf.split(layers[2], 2, 0)
	size = tf.size(gen_f)
	style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

	gen_f, style_f = tf.split(layers[3], 2, 0)
	size = tf.size(gen_f)
	style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

	return style_loss

def gram(layer):
	shape = tf.shape(layer)
	num_images = shape[0]
	width = shape[1]
	height = shape[2]
	num_filters = shape[3]
	filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
	# grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

	return filters

def makeup_loss(I_src_B, histogram, lambda_list, length):
	loss = 0
	size = 0
	for i in range(0, length):
		size = tf.count_nonzero(I_src_B)
		difference = I_src_B[i] - histogram[i]
		difference = tf.square(difference)
		total = tf.reduce_sum(difference)
		loss += total * lambda_list[i]
		size = tf.cast(size, tf.float32)
	return loss / size * 3.0