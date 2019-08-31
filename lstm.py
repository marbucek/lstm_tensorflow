import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LSTM():

    def __init__(self, FLAGS, train_embeddings = True, down_project = False):

        self.FLAGS = FLAGS
        self.train_embeddings = train_embeddings
        self.down_project = down_project
        self._initialize_weights()

    def _initialize_weights(self):

        self.V = self.FLAGS.vocab_size
        self.E = self.FLAGS.embed_size
        self.H = self.FLAGS.hidden_size
        self.P = self.FLAGS.projection_size

        with tf.variable_scope('embedding', reuse = tf.AUTO_REUSE):
            self.W_embedding = tf.get_variable('W_embedding', [self.V, self.E], trainable = self.train_embeddings,
                                                initializer = xavier())

        with tf.variable_scope('lstm_cell', reuse = tf.AUTO_REUSE):
            self.W_fx = tf.get_variable('W_forget_x', [self.E, self.H], initializer = xavier())
            self.W_fh = tf.get_variable('W_forget_h', [self.H, self.H], initializer = xavier())
            self.b_f = tf.get_variable('b_forget', [self.H], initializer = xavier())

            self.W_ix = tf.get_variable('W_input_x', [self.E, self.H], initializer = xavier())
            self.W_ih = tf.get_variable('W_input_h', [self.H, self.H], initializer = xavier())
            self.b_i = tf.get_variable('b_input', [self.H], initializer = xavier())

            self.W_cx = tf.get_variable('W_candidate_x', [self.E, self.H], initializer = xavier())
            self.W_ch = tf.get_variable('W_candidate_h', [self.H, self.H], initializer = xavier())
            self.b_c = tf.get_variable('b_candidate', [self.H], initializer = xavier())

            self.W_ox = tf.get_variable('W_output_x', [self.E, self.H], initializer = xavier())
            self.W_oh = tf.get_variable('W_output_h', [self.H, self.H], initializer = xavier())
            self.b_o = tf.get_variable('b_output', [self.H], initializer = xavier())

        if self.down_project == False:
            with tf.variable_scope('softmax', reuse = tf.AUTO_REUSE):
                self.W_softmax = tf.get_variable('W_softmax', [self.H, self.V], initializer=xavier())
                self.b_softmax = tf.get_variable('b_softmax', [self.V], initializer=xavier())
        else:
            with tf.variable_scope('projection', reuse = tf.AUTO_REUSE):
                self.W_down_proj = tf.get_variable('W_down_proj', [self.H, self.P], initializer=xavier())
                self.b_down_proj = tf.get_variable('b_down_proj', [self.P], initializer=xavier())

            with tf.variable_scope('softmax', reuse = tf.AUTO_REUSE):
                self.W_softmax = tf.get_variable('W_softmax', [self.P, self.V], initializer=xavier())
                self.b_softmax = tf.get_variable('b_softmax', [self.V], initializer=xavier())


    def _lstm_cell(self, lstm_t_minus_1, x_t):

        h_t_minus_1, c_t_minus_1 = tf.unstack(lstm_t_minus_1)

        f_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_fx) + tf.matmul(h_t_minus_1, self.W_fh) + self.b_f)
        i_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_ix) + tf.matmul(h_t_minus_1, self.W_ih) + self.b_i)
        c_hat_t = tf.nn.tanh(tf.matmul(x_t, self.W_cx) + tf.matmul(h_t_minus_1, self.W_ch) + self.b_c)
        c_t = (f_t * c_t_minus_1) + (i_t * c_hat_t)
        o_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_ox) + tf.matmul(h_t_minus_1, self.W_oh) + self.b_o)
        h_t = o_t * tf.nn.tanh(c_t)

        lstm_t = tf.stack([h_t, c_t])

        return lstm_t

    def build_graph(self):

        self.input = tf.placeholder(tf.int64, [None, self.FLAGS.sentence_length], name="input")
        self.target = tf.placeholder(tf.int64, [None, self.FLAGS.sentence_length], name="target")
        self.h_0 = tf.placeholder(tf.float32, [None, self.H], name='hidden_state_initial')
        self.c_0 = tf.placeholder(tf.float32, [None, self.H], name='current_state_initial')
        self.lstm_0 = tf.stack([self.h_0, self.c_0])

        lstm_t = self.lstm_0
        x_input = tf.nn.embedding_lookup(self.W_embedding,self.input) #embedded input
        logits = []
        for t in range(self.FLAGS.sentence_length):
            lstm_t = self._lstm_cell(lstm_t, x_input[:, t, :])
            h_t, c_t = tf.unstack(lstm_t)

            if self.down_project == False:
                logits.append(tf.matmul(h_t, self.W_softmax) + self.b_softmax)
            else:
                h_t_tilde = tf.matmul(h_t, self.W_down_proj) + self.b_down_proj
                logits.append(tf.matmul(h_t_tilde, self.W_softmax) + self.b_softmax)

        self.logits = tf.stack(logits, axis = 1)
        self.pred = tf.argmax(self.logits, axis = -1)

        #for evaluating the loss, compare 2nd -> last element of the target to the 1st -> 2nd-to-last element of the predictions
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = self.target[:,1:],
                logits = self.logits[:,:-1,:])
        self.loss = tf.reduce_mean(tf.reduce_sum(self.cross_entropy, axis = -1))

    def build_graph_predictions(self):

        self.input_word = tf.placeholder(tf.int64, [None], name = 'input_word')
        self.lstm_t_minus_1 = tf.placeholder(tf.float32,[2, None, self.H], name = 'lstm_state_previous')

        x_t = tf.nn.embedding_lookup(self.W_embedding, self.input_word)
        self.lstm_t = self._lstm_cell(self.lstm_t_minus_1, x_t)
        h_t, c_t = tf.unstack(self.lstm_t)

        if self.down_project == False:
            logits = tf.matmul(h_t, self.W_softmax) + self.b_softmax
        else:
            h_t_tilde = tf.matmul(h_t, self.W_down_proj) + self.b_down_proj
            self.logits = tf.matmul(h_t_tilde, self.W_softmax) + self.b_softmax

        self.pred = tf.argmax(self.logits, axis = -1)


    def predict_next_token(self, sess, lstm_t_minus_1, input_word):

        lstm_t, prediction, logits = sess.run([self.lstm_t, self.pred, self.logits], feed_dict = {self.lstm_t_minus_1: lstm_t_minus_1, self.input_word: input_word})
        return lstm_t, prediction, logits

    def build_optimizer(self):

        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            self.optimizer = tf.train.AdamOptimizer()
            self.grads = tf.gradients(self.loss, tf.trainable_variables())
            self.clipped_grads, _ = tf.clip_by_global_norm(self.grads, self.FLAGS.clip_norm)
            self.train_step = self.optimizer.apply_gradients(zip(self.clipped_grads, tf.trainable_variables()))

    def evaluate(self, sess, input):

        loss, cross_entropy = sess.run([self.loss, self.cross_entropy], feed_dict = {
                                                self.input: input,
                                                self.target: input,
                                                self.h_0: np.zeros([input.shape[0],self.H]),
                                                self.c_0: np.zeros([input.shape[0],self.H])
                                                })
        return loss, cross_entropy


    def training_step(self, sess, input, target):

        _, loss = sess.run([self.train_step, self.loss] , feed_dict = {
                                                self.input: input,
                                                self.target: target,
                                                self.h_0: np.zeros([input.shape[0],self.H]),
                                                self.c_0: np.zeros([input.shape[0],self.H])
                                                })

        return loss
