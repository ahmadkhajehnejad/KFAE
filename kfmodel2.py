from __future__ import absolute_import, division, print_function

import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf

class KFModel2(object):

    def __init__(self, S_dim, O_dim, h_dim, batch_size, learning_rate, log_dir):
        
        self.S_t_minus_1 = tf.placeholder(tf.float32, [None, S_dim])
        self.S_t = tf.placeholder(tf.float32, [None, S_dim])
        self.S_p_t_placeholder = tf.placeholder(tf.float32, [None, h_dim])
        self.O_t = tf.placeholder(tf.float32, [None, O_dim])
        self.O_p_t_placeholder = tf.placeholder(tf.float32, [None, h_dim])
        
        self.A_1 = tf.get_variable('A_1', initializer=tf.eye(h_dim))
        self.A_2 = tf.get_variable('A_2', initializer=tf.eye(h_dim))
        self.b_1 = tf.get_variable('b_1', initializer=tf.zeros([h_dim,1]))
        self.b_2 = tf.get_variable('b_2', initializer=tf.zeros([h_dim,1]))
        self.R_1 = tf.get_variable('R_1', initializer=tf.eye(h_dim) * 0.1)
        self.R_2 = tf.get_variable('R_2', initializer=tf.eye(h_dim) * 0.1)
        
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
            
            self.eps_1 = tf.random_normal([batch_size, h_dim], stddev=1.)
            
            self.S_p_bar_t = tf.matmul(self.S_p_t_minus_1, tf.transpose(self.A_1)) + \
                             tf.matmul(self.eps_1, tf.transpose(self.R_1))
            
            with tf.variable_scope("decoder_S"):
                self.S_tilde_t = decoder(self.S_p_bar_t, S_dim)
            with tf.variable_scope("decoder_S", reuse=True):
                self.S_bar_t_minus_1 = decoder(self.S_p_t_minus_1, S_dim)
            with tf.variable_scope("decoder_S", reuse=True):
                self.S_t_decoded = decoder(self.S_p_t_placeholder, S_dim)
            
            self.eps_2 = tf.random_normal([batch_size, h_dim], stddev=1.)
            self.O_p_bar_t = tf.matmul(self.S_p_t, tf.transpose(self.A_2)) + \
                             tf.matmul(self.eps_2, tf.transpose(self.R_2))
            with tf.variable_scope("decoder_O"):
                self.O_tilde_t = decoder(self.O_p_bar_t, O_dim)
                
            with tf.variable_scope("encoder_O"):
                self.O_p_t = encoder(self.O_t, h_dim)
            with tf.variable_scope("decoder_O", reuse=True):
                self.O_bar_t = decoder(self.O_p_t, O_dim)
            with tf.variable_scope("decoder_O", reuse=True):
                self.O_t_decoded = decoder(self.O_p_t_placeholder, O_dim)
            
            self.Y_1 = self.S_p_t - tf.matmul(self.S_p_t_minus_1, tf.transpose(self.A_1))
            self.Y_2 = self.O_p_t - tf.matmul(self.S_p_t, tf.transpose(self.A_2))
            self.pos_samples_1 = tf.matmul(tf.random_normal([batch_size, h_dim], stddev=1.), tf.transpose(self.R_1))
            self.pos_samples_2 = tf.matmul(tf.random_normal([batch_size, h_dim], stddev=1.), tf.transpose(self.R_2))

            with tf.variable_scope('discriminator'):
                with tf.variable_scope('D1'):
                    self.pos_samples_1_pred = discriminator(self.pos_samples_1)
                with tf.variable_scope('D1', reuse=True):
                    self.neg_samples_1_pred = discriminator(self.Y_1)
                with tf.variable_scope('D2'):
                    self.pos_samples_2_pred = discriminator(self.pos_samples_2)
                with tf.variable_scope('D2', reuse=True):
                    self.neg_samples_2_pred = discriminator(self.Y_2)
            
            self.S_recons_loss = tf.reduce_mean(tf.norm(self.S_bar_t_minus_1 - self.S_t_minus_1, axis=1))
            self.S_pred_loss = tf.reduce_mean(tf.norm(self.S_tilde_t - self.S_t, axis=1))
            self.O_recons_loss = tf.reduce_mean(tf.norm(self.O_bar_t - self.O_t, axis=1))
            self.O_pred_loss = tf.reduce_mean(tf.norm(self.O_tilde_t - self.O_t, axis=1))
            self.D1_cross_ent = compute_classification_loss(self.pos_samples_1_pred, self.neg_samples_1_pred)
            self.D2_cross_ent = compute_classification_loss(self.pos_samples_2_pred, self.neg_samples_2_pred)
            
            # add summary ops
            tf.summary.scalar('S_recons_loss', self.S_recons_loss)
            tf.summary.scalar('S_pred_loss', self.S_pred_loss)
            tf.summary.scalar('O_recons_loss', self.O_recons_loss)
            tf.summary.scalar('O_pred_loss', self.O_pred_loss)
            tf.summary.scalar('D1_cross_ent', self.D1_cross_ent)
            tf.summary.scalar('D2_cross_ent', self.D2_cross_ent)

            # define references to params
            encoder_S_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_S')
            decoder_S_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder_S')
            encoder_O_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_O')
            decoder_O_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder_O')
            Gauss_S_params = [self.A_1, self.b_1, self.R_1]
            Gauss_O_params = [self.A_2, self.b_2, self.R_2]
            D1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator/D1')
            D2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator/D2')
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            # define training steps
            self.learn_rate = self._get_learn_rate(global_step, learning_rate)
            
            # S_recons_loss optimizer
            self.optimize_S_recons_loss = layers.optimize_loss(self.S_recons_loss, \
                    global_step, self.learn_rate * 1, optimizer=lambda lr: \
                    #tf.train.GradientDescentOptimizer(lr), variables=\
                    tf.train.AdamOptimizer(lr), variables=\
                    #tf.train.MomentumOptimizer(lr, 0.1), variables=\
                    encoder_S_params + decoder_S_params, update_ops=[])
            
            self.optimize_S_pred_loss = layers.optimize_loss(self.S_pred_loss, \
                    global_step, self.learn_rate * 1, optimizer=lambda lr: \
                    tf.train.AdamOptimizer(lr), variables=\
                    #tf.train.MomentumOptimizer(lr, 0.1), variables=\
                    #Gauss_S_params, update_ops=[])
                    Gauss_S_params + encoder_S_params + decoder_S_params, update_ops=[])
            
            self.optimize_O_recons_loss = layers.optimize_loss(self.O_recons_loss, \
                    global_step, self.learn_rate * 1, optimizer=lambda lr: \
                    #tf.train.GradientDescentOptimizer(lr), variables=\
                    tf.train.AdamOptimizer(lr), variables=\
                    #tf.train.MomentumOptimizer(lr, 0.1), variables=\
                    encoder_O_params + decoder_O_params, update_ops=[])
            
            self.optimize_O_pred_loss = layers.optimize_loss(self.O_pred_loss, \
                    global_step, self.learn_rate * 1, optimizer=lambda lr: \
                    tf.train.AdamOptimizer(lr), variables=\
                    #tf.train.MomentumOptimizer(lr, 0.1), variables=\
                    #Gauss_O_params, update_ops=[])
                    Gauss_O_params + encoder_S_params + decoder_O_params, update_ops=[])
            
            self.minimize_D1_cross_ent = layers.optimize_loss(self.D1_cross_ent, \
                    global_step, self.learn_rate * 10, optimizer=lambda lr: \
                    #tf.train.AdamOptimizer(lr), variables=\
                    tf.train.MomentumOptimizer(lr, 0.1), variables=\
                    D1_params, update_ops=[])
            
            self.maximize_D1_cross_ent = layers.optimize_loss(-self.D1_cross_ent, \
                    global_step, self.learn_rate, optimizer=lambda lr: \
                    tf.train.AdamOptimizer(lr), variables=\
                    #tf.train.MomentumOptimizer(lr, 0.9), variables=\
                    encoder_S_params, update_ops=[])
            
            self.minimize_D2_cross_ent = layers.optimize_loss(self.D2_cross_ent, \
                    global_step, self.learn_rate * 10, optimizer=lambda lr: \
                    #tf.train.AdamOptimizer(lr), variables=\
                    tf.train.MomentumOptimizer(lr, 0.1), variables=\
                    D2_params, update_ops=[])
            
            self.maximize_D2_cross_ent = layers.optimize_loss(-self.D2_cross_ent, \
                    global_step, self.learn_rate, optimizer=lambda lr: \
                    tf.train.AdamOptimizer(lr), variables=\
                    #tf.train.MomentumOptimizer(lr, 0.9), variables=\
                    encoder_S_params + encoder_O_params, update_ops=[])

            
            self.sess = tf.Session()
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(log_dir, \
                self.sess.graph)
            self.sess.run(tf.global_variables_initializer())
            
    def _get_learn_rate(self, global_step, learning_rate):
        
        boundaries = [np.int64(1000000)]
        values = [learning_rate, learning_rate/10]
        
        return tf.train.piecewise_constant(global_step, boundaries, values)

        
    def update_params(self, input_S_t_minus_1, input_S_t, input_O_t):
        _S_recons_loss, _S_pred_loss, _O_recons_loss, _O_pred_loss,\
        _D1_cross_ent, _D2_cross_ent = 0, 0, 0, 0, 0, 0
        summary = []
        
        _S_recons_loss = self.sess.run(self.optimize_S_recons_loss,\
                               {self.S_t_minus_1: input_S_t_minus_1}
                              )
        _S_pred_loss = self.sess.run(self.optimize_S_pred_loss,\
                               {self.S_t_minus_1: input_S_t_minus_1, self.S_t: input_S_t}
                              )
        
        _O_recons_loss = self.sess.run(self.optimize_O_recons_loss,\
                               {self.O_t: input_O_t}
                              )
        _O_pred_loss = self.sess.run(self.optimize_O_pred_loss,\
                               {self.S_t: input_S_t, self.O_t: input_O_t}
                              )
        '''
        _D1_cross_ent = self.sess.run(self.minimize_D1_cross_ent,\
                               {self.S_t_minus_1: input_S_t_minus_1, self.S_t: input_S_t}
                              )
        _ = self.sess.run(self.maximize_D1_cross_ent,\
                               {self.S_t_minus_1: input_S_t_minus_1, self.S_t: input_S_t}
                              )
        _D2_cross_ent = self.sess.run(self.minimize_D2_cross_ent,\
                               {self.S_t: input_S_t, self.O_t: input_O_t}
                              )
        
        summary, _ = self.sess.run([self.merged, self.maximize_D2_cross_ent],\
                                          {self.S_t_minus_1: input_S_t_minus_1, self.S_t: input_S_t,\
                                           self.O_t: input_O_t\
                                          }\
                                         )
        '''
        summary = self.sess.run(self.merged,\
                                          {self.S_t_minus_1: input_S_t_minus_1, self.S_t: input_S_t,\
                                           self.O_t: input_O_t\
                                          }\
                                         )
        
        
        return _S_recons_loss, _S_pred_loss, _O_recons_loss, _O_pred_loss, \
               _D1_cross_ent, _D2_cross_ent, summary

    def decode_O_p_t(self,input_O_p_t):
        return self.sess.run(self.O_t_decoded, {self.O_p_t_placeholder: input_O_p_t})
    
    def decode_S_p_t(self,input_S_p_t):
        return self.sess.run(self.S_t_decoded, {self.S_p_t_placeholder: input_S_p_t})
        
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

def discriminator(input_tensor):
    return encoder(input_tensor, 1)

def compute_classification_loss(pos_logit, neg_logit):
    return tf.losses.sigmoid_cross_entropy(\
                    tf.ones(tf.shape(pos_logit)), pos_logit) +\
                    tf.losses.sigmoid_cross_entropy(tf.zeros(\
                    tf.shape(neg_logit)), neg_logit)
