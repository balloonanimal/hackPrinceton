from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import time
import functools
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

# Main TFGAN library.
tfgan = tf.contrib.gan

# Shortcuts for later.
queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def visualize_training_generator(train_step_num, start_time, data_np):
    """Visualize generator outputs during training.
    
    Args:
        train_step_num: The training step number. A python integer.
        start_time: Time when training started. The output of `time.time()`. A
            python float.
        data: Data to plot. A numpy array, most likely from an evaluated TensorFlow
            tensor.
    """
    print('Training step: %i' % train_step_num)
    time_since_start = (time.time() - start_time) / 60.0
    print('Time since start: %f m' % time_since_start)
    print('Steps per min: %f' % (train_step_num / time_since_start))
    plt.axis('off')
    plt.imshow(np.squeeze(data_np), cmap='gray')
    plt.show()

def visualize_digits(tensor_to_visualize):
    """Visualize an image once. Used to visualize generator before training.
    
    Args:
        tensor_to_visualize: An image tensor to visualize. A python Tensor.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with queues.QueueRunners(sess):
            images_np = sess.run(tensor_to_visualize)
            plt.axis('off')
            plt.imshow(np.squeeze(images_np), cmap='gray')
            plt.show()

def evaluate_tfgan_loss(gan_loss, name=None):
    """Evaluate GAN losses. Used to check that the graph is correct.
    
    Args:
        gan_loss: A GANLoss tuple.
        name: Optional. If present, append to debug output.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with queues.QueueRunners(sess):
            gen_loss_np = sess.run(gan_loss.generator_loss)
            dis_loss_np = sess.run(gan_loss.discriminator_loss)
            if name:
                print('%s generator loss: %f' % (name, gen_loss_np))
                print('%s discriminator loss: %f'% (name, dis_loss_np))
            else:
                print('Generator loss: %f' % gen_loss_np)
                print('Discriminator loss: %f'% dis_loss_np)

from provider import provider
data_provider = provider()

tf.reset_default_graph()

# Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
# for forward and backwards propogation.
batch_size = 32
with tf.device('/cpu:0'):
    real_images, one_hot_labels = data_provider.provide_data(
        'train', batch_size)

    # Sanity check that we're getting images.
    check_real_digits = tfgan.eval.image_reshaper(real_images[:20,...], num_cols=10)
    # visualize_digits(check_real_digits)

def conditional_generator_fn(inputs, weight_decay=2.5e-5, is_training=True):
        """Generator to produce MNIST images.
    
    Args:
        inputs: A 2-tuple of Tensors (noise, one_hot_labels).
        weight_decay: The value of the l2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population 
            statistics.

    Returns:
        A generated image in the range [-1, 1].
    """
        noise, one_hot_labels = inputs
        with framework.arg_scope(
                [layers.fully_connected, layers.conv2d_transpose],
                activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                weights_regularizer=layers.l2_regularizer(weight_decay)),\
                framework.arg_scope([layers.batch_norm], is_training=is_training,
                                    zero_debias_moving_mean=True):
            net = layers.fully_connected(noise, 1024)
            net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
            net = layers.fully_connected(net, 7 * 7 * 128)
            net = tf.reshape(net, [-1, 7, 7, 128])
            net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
            net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
            # Make sure that generator output is in the same range as `inputs`
            # ie [-1, 1].
            net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)

            return net

def conditional_discriminator_fn(img, conditioning, weight_decay=2.5e-5):
        """Conditional discriminator network on MNIST digits.
    
    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        conditioning: A 2-tuple of Tensors representing (noise, one_hot_labels).
        weight_decay: The L2 weight decay.

    Returns:
        Logits for the probability that the image is real.
    """
        _, one_hot_labels = conditioning
        with tf.device('/gpu:0'):
            with framework.arg_scope(
                    [layers.conv2d, layers.fully_connected],
                    activation_fn=leaky_relu, normalizer_fn=None,
                    weights_regularizer=layers.l2_regularizer(weight_decay),
                    biases_regularizer=layers.l2_regularizer(weight_decay)):
                net = layers.conv2d(img, 64, [4, 4], stride=2)
                net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
                net = layers.conv2d(net, 128, [4, 4], stride=2)
                net = layers.flatten(net)
                net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)

                return layers.linear(net, 1)

noise_dims = 64
with tf.device('/gpu:0'):
    conditional_gan_model = tfgan.gan_model(
        generator_fn=conditional_generator_fn,
        discriminator_fn=conditional_discriminator_fn,
        real_data=real_images,
        generator_inputs=(tf.random_normal([batch_size, noise_dims]),
                                                one_hot_labels))
    
print(conditional_gan_model.generated_data)

# Sanity check that currently generated images are garbage.
cond_generated_data_to_visualize = tfgan.eval.image_reshaper(
        conditional_gan_model.generated_data[:20,...], num_cols=10)
# visualize_digits(cond_generated_data_to_visualize)

gan_loss = tfgan.gan_loss(
        conditional_gan_model, gradient_penalty_weight=1.0)

# Sanity check that we can evaluate our losses.
# evaluate_tfgan_loss(gan_loss)

with tf.device('/gpu:0'):
    generator_optimizer = tf.train.AdamOptimizer(0.0009, beta1=0.5)
    discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)
    gan_train_ops = tfgan.gan_train_ops(
        conditional_gan_model,
        gan_loss,
        generator_optimizer,
        discriminator_optimizer)


# Set up class-conditional visualization. We feed class labels to the generator
# so that the the first column is `0`, the second column is `1`, etc.

images_to_eval = 500
assert images_to_eval % 10 == 0
with tf.device('/gpu:0'):
    random_noise = tf.random_normal([images_to_eval, 64])
    one_hot_labels = tf.one_hot(
        [i for _ in xrange(images_to_eval // 10) for i in xrange(10)], depth=26)

    print(type(one_hot_labels))

    with tf.variable_scope('Generator', reuse=True):
        eval_images = conditional_gan_model.generator_fn(
            (random_noise, one_hot_labels), is_training=False)

        print(type(eval_images))
        print(type(conditional_gan_model.generator_fn))

        reshaped_eval_imgs = tfgan.eval.image_reshaper(
            eval_images[:20, ...], num_cols=10)
        
        print(tf.shape(reshaped_eval_imgs))
        global_step = tf.train.get_or_create_global_step()
        train_step_fn = tfgan.get_sequential_train_steps()
        loss_values  = []

# saver = tf.train.Saver()

# from itertools import count

# with tf.train.SingularMonitoredSession() as sess:
#     start_time = time.time()
#     for i in count():
#         cur_loss, _ = train_step_fn(
#             sess, gan_train_ops, global_step, train_step_kwargs={})
#         loss_values.append((i, cur_loss))
#         if i % 10000 == 0:
#             digits_np = sess.run([reshaped_eval_imgs])
#             print('Current loss: %f' % cur_loss)
#             # visualize_training_generator(i, start_time, digits_np)
    
#             save_path = saver.save(sess.raw_session(), "./checkpoints/model.ckpt")
#             print("Model saved in path: %s" % save_path)
