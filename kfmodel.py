from __future__ import absolute_import, division, print_function

import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf

class KFModel(object):

    def __init__(self, S_dim, O_dim, h_dim, batch_size, learning_rate, log_dir):
        
        self.S_t_minus_1 = tf.placeholder(tf.float32, [None, S_dim])
        self.S_t = tf.placeholder(tf.float32, [None, S_dim])
        self.O_t = tf.placeholder(tf.float32, [None, O_dim])
        self.O_t_bar_placeholder = tf.placeholder(tf.float32, [None, O_dim])
        
        self.A_1 = tf.get_variable('A_1', initializer=tf.eye(h_dim))
        self.A_2 = tf.get_variable('A_2', initializer=tf.eye(h_dim))
        self.b_1 = tf.get_variable('b_1', initializer=tf.zeros([h_dim,1]))
        self.b_2 = tf.get_variable('b_2', initializer=tf.zeros([h_dim,1]))
        self.R_1 = tf.get_variable('R_1', initializer=tf.eye(h_dim))
        self.R_2 = tf.get_variable('R_2', initializer=tf.eye(h_dim))
        
        #self.A_1 = tf.get_variable('A_1', [h_dim, h_dim], initializer=tf.random_normal_initializer(stddev=np.sqrt(1)))
        #self.A_2 = tf.get_variable('A_2', [h_dim, h_dim], initializer=tf.random_normal_initializer(stddev=np.sqrt(1)))
        #self.b_1 = tf.get_variable('b_1', [h_dim, 1], initializer=tf.random_normal_initializer(stddev=np.sqrt(1)))
        #self.b_2 = tf.get_variable('b_2', [h_dim, 1], initializer=tf.random_normal_initializer(stddev=np.sqrt(1)))
        #self.R_1 = tf.get_variable('R_1', [h_dim, h_dim], initializer=tf.random_normal_initializer(stddev=np.sqrt(1)))
        #self.R_2 = tf.get_variable('R_2', [h_dim, h_dim], initializer=tf.random_normal_initializer(stddev=np.sqrt(1)))
        
        
        with arg_scope([layers.fully_connected], activation_fn=tf.nn.relu):
            with tf.variable_scope("encoder_S"):
                self.S_p_t_minus_1 = encoder(self.S_t_minus_1, h_dim)
            with tf.variable_scope("encoder_S", reuse=True):
                self.S_p_t = encoder(self.S_t, h_dim)
            
            self.noise_1 = tf.random_normal([batch_size, h_dim], stddev=1.)
            
            self.S_p_t_bar = tf.matmul(self.S_p_t_minus_1, tf.transpose(self.A_1)) + \
                         tf.matmul(self.noise_1, tf.transpose(self.R_1))
            
            with tf.variable_scope("decoder_S"):
                self.S_t_bar = decoder(self.S_p_t_bar, S_dim)
            
            self.noise_2 = tf.random_normal([batch_size, h_dim], stddev=1.)
            self.O_p_t_bar = tf.matmul(self.S_p_t, tf.transpose(self.A_2)) + \
                             tf.matmul(self.noise_2, tf.transpose(self.R_2))
            with tf.variable_scope("decoder_O"):
                self.O_t_bar = decoder(self.O_p_t_bar, O_dim)
            with tf.variable_scope("encoder_O"):
                self.O_p_t_tilde = encoder(self.O_t_bar, h_dim)
            with tf.variable_scope("encoder_O", reuse=True):
                self.O_p_t_tilde_2 = encoder(self.O_t_bar_placeholder, h_dim)
            
            S_t_recons_loss = tf.reduce_mean(tf.norm(self.S_t_bar - self.S_t, axis=1))
            O_t_recons_loss = tf.reduce_mean(tf.norm(self.O_t_bar - self.O_t, axis=1))
            O_p_t_recons_loss = tf.reduce_mean(tf.norm(self.O_p_t_tilde - self.O_p_t_bar, axis=1))
            
            # add summary ops
            tf.summary.scalar('S_t_recons_loss', S_t_recons_loss)
            tf.summary.scalar('O_t_recons_loss', O_t_recons_loss)
            tf.summary.scalar('O_p_t_recons_loss', O_p_t_recons_loss)

            # define references to params
            encoder_S_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_S')
            decoder_S_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder_S')
            encoder_O_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_O')
            decoder_O_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder_O')
            Gauss_S_params = [self.A_1, self.b_1, self.R_1]
            Gauss_O_params = [self.A_2, self.b_2, self.R_2]
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            # define training steps
            self.learn_rate = self._get_learn_rate(global_step, learning_rate)
            
            # update autoencoder params to minimise reconstruction loss
            self.train_S_params = layers.optimize_loss(S_t_recons_loss, \
                    global_step, self.learn_rate, optimizer=lambda lr: \
                    tf.train.AdamOptimizer(lr), variables=\
                    #tf.train.MomentumOptimizer(lr, 0.9), variables=\
                    encoder_S_params + Gauss_S_params + decoder_S_params, update_ops=[])
            
            self.train_O_params = layers.optimize_loss(O_t_recons_loss, \
                    global_step, self.learn_rate, optimizer=lambda lr: \
                    tf.train.AdamOptimizer(lr), variables=\
                    #tf.train.MomentumOptimizer(lr, 0.9), variables=\
                    Gauss_O_params + decoder_O_params, update_ops=[])
            
            
            self.train_O_p_params = layers.optimize_loss(O_p_t_recons_loss, \
                    global_step, self.learn_rate, optimizer=lambda lr: \
                    tf.train.AdamOptimizer(lr), variables=\
                    #tf.train.MomentumOptimizer(lr, 0.9), variables=\
                    encoder_S_params + Gauss_S_params + decoder_S_params + \
                    Gauss_O_params + decoder_O_params + encoder_O_params , update_ops=[])
            
            self.sess = tf.Session()
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(log_dir, \
                self.sess.graph)
            self.sess.run(tf.global_variables_initializer())
            
    def _get_learn_rate(self, global_step, learning_rate):
        
        boundaries = [np.int64(100000)]
        values = [learning_rate, learning_rate/10]
        
        return tf.train.piecewise_constant(global_step, boundaries, values)

        
    def update_params(self, input_S_t_minus_1, input_S_t, input_O_t):
        S_loss = self.sess.run(self.train_S_params,\
                               {self.S_t_minus_1: input_S_t_minus_1, self.S_t: input_S_t}
                              )
        O_loss = self.sess.run(self.train_O_params,\
                               {self.S_t: input_S_t, self.O_t: input_O_t}
        )
        #O_loss = 0
        
        summary, O_p_loss = self.sess.run([self.merged, self.train_O_p_params],\
                                          {self.S_t_minus_1: input_S_t_minus_1, self.S_t: input_S_t,\
                                           self.O_t: input_O_t\
                                          }\
                                         )
        
        #O_p_loss = 0
        #summary = self.sess.run(self.merged,\
        #                        {self.S_t_minus_1: input_S_t_minus_1, self.S_t: input_S_t,\
        #                         self.O_t: input_O_t\
        #                        }\
        #                       )
        
        return S_loss, O_loss, O_p_loss, summary
    
    def encoded_S(self, input_S):
        return self.sess.run(self.S_p_t_minus_1, {self.S_t_minus_1: input_S})
    
    def encoded_O(self, input_O):
        return self.sess.run(self.O_p_t_tilde_2, {self.O_p_t_placeholder: input_O})
    
def encoder(input_tensor, output_size):
    net = layers.fully_connected(input_tensor, 40)
    net = layers.fully_connected(net, 60)
    net = layers.fully_connected(net, 100)
    net = layers.fully_connected(net, output_size, activation_fn=None)
    return net

def decoder(input_tensor, output_size):
    net = layers.fully_connected(input_tensor, 100)
    net = layers.fully_connected(net, 60)
    net = layers.fully_connected(net, 40)
    net = layers.fully_connected(net, output_size, activation_fn=None)
    return net

