from lstm import LSTM
import numpy as np
import os, pickle, shutil
import tensorflow as tf
from utils import word2index, index2word, preprocess, perplexity
from load_embedding import load_embedding
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FILE_TRAIN = 'data/sentences.train'
FILE_TEST = 'data/sentences_test2.txt'
FILE_EVAL = 'data/sentences.eval'
FILE_CONT = 'data/sentences.continuation'

GROUP_NUMBER = 32

class batch_iterator():
    """Returns the next batch of 64 (possibly randomly chosen) lines in the file"""

    def __init__(self, FLAGS, file, random = True):
        self.batch_size = FLAGS.batch_size
        self.sentence_length = FLAGS.sentence_length
        self.random = random

        #selecting only rows of length up to 28 such that the total length with <bos> and <eos> does not exceed 30
        self.rows = np.array([row.split() for row in file if len(row.split()) <= self.sentence_length - 2])
        self.row_ids = np.random.permutation(len(self.rows))

        self.i = 0
        self.iteration = 0
        self.epoch = 0
        self.completed_epoch = False

    def __iter__(self):

        return self

    def __next__(self):

        i = self.i

        if i + self.batch_size >= len(self.rows):
            self.row_ids = np.random.permutation(len(self.rows))
            self.i = 0
            self.epoch += 1
            self.completed_epoch = True

            if self.random == True:
                self.return_ids = self.row_ids[i:]
            else:
                self.return_ids = np.array(range(i,len(self.rows)))

        else:
            self.completed_epoch = False
            self.i = self.i + self.batch_size

            if self.random == True:
                self.return_ids = self.row_ids[i:i+self.batch_size]
            else:
                self.return_ids = np.array(range(i,i+self.batch_size))


        sentences = self.rows[self.return_ids]
        self.iteration += 1

        return sentences

def continuation(FLAGS, input_sentences = None, use_unknown = True):

    print('Building LSTM')
    model = LSTM(FLAGS, down_project = FLAGS.down_project)
    model.build_graph_predictions()

    print('Loading word2index')
    w2i = pickle.load(
                open(os.path.join(FLAGS.out_dir,'w2i'),'rb')
                )

    if input_sentences == None:
        print('Opening input file')
        file =  open(FILE_CONT,'r')
        sentences = file.readlines()
        file_output = open(os.path.join(FLAGS.out_dir,'group{}.continuation'.format(GROUP_NUMBER)),'w')
    else:
        sentences = input_sentences

    with tf.Session() as sess:

        saver = tf.train.Saver()
        load_dir = os.path.join(FLAGS.out_dir,FLAGS.experiment)
        last_checkpoint = tf.train.latest_checkpoint(load_dir)
        print('Restoring weights from the checkpoint {}'.format(last_checkpoint))
        saver.restore(sess, last_checkpoint)

        print('Computing the continuations\n')
        for sentence in sentences:
            sentence = sentence.split()
            sentence.insert(0,'<bos>')
            input = word2index(sentence, w2i)
            output = []

            lstm_t = np.stack([np.zeros([1,FLAGS.hidden_size]),np.zeros([1,FLAGS.hidden_size])])
            for t in range(FLAGS.sentence_cont_length + 1): #generate sentences of length up to 20 excluding the <bos> symbol
                if t < len(input):
                    word_t = np.array([input[t]])

                else:
                    word_t = out_word_t
                output.append(word_t[0])
                if word_t == 1: #if we hit <eos>
                    break
                lstm_t, out_word_t, logits = model.predict_next_token(sess, lstm_t, word_t)
                if out_word_t == 2 and not use_unknown: #if not using unknown, use the 2nd most probable word
                    out_word_t = np.argsort(logits, axis = 1)[:,-2]
            output = " ".join(index2word(output[1:],w2i)) #convert output to sentence omitting the <bos> symbol

            if input_sentences == None:
                file_output.write('{}\n'.format(output))
            else:
                print(output)


def evaluate(FLAGS):

    print('Building LSTM')
    model = LSTM(FLAGS, down_project = FLAGS.down_project)
    model.build_graph()

    print('Loading word2index')
    w2i = pickle.load(
                open(os.path.join(FLAGS.out_dir,'w2i'),'rb')
                )

    print('Opening testing file')
    with open(FILE_TEST,'r') as file:
        sentences_test = batch_iterator(FLAGS, file, random = False)

    file_output = open(os.path.join(FLAGS.out_dir,'group{}.perplexity{}'.format(GROUP_NUMBER,FLAGS.experiment)),'w')

    with tf.Session() as sess:

        saver = tf.train.Saver()
        load_dir = os.path.join(FLAGS.out_dir,FLAGS.experiment)
        last_checkpoint = tf.train.latest_checkpoint(load_dir)
        print('Restoring weights from the checkpoint {}'.format(last_checkpoint))
        saver.restore(sess, last_checkpoint)

        print('Evaluating on the testing file')
        sentences_processed = 0

        while sentences_test.completed_epoch == 0:
            input_test = np.stack([word2index(preprocess(sentence, max_length = FLAGS.sentence_length), w2i) for sentence in next(sentences_test)])
            loss, cross_entropy = model.evaluate(sess, input = input_test)
            perplexities = perplexity(cross_entropy, input_test)
            sentences_processed += len(perplexities)

            for perp in perplexities:
                file_output.write('{}\n'.format(perp))

    print('Wrote {} perplexities into the file group{}.perplexity{}.'.format(sentences_processed,GROUP_NUMBER,FLAGS.experiment))


def train(FLAGS):

    print('Opening training file')
    with open(FILE_TRAIN,'r') as file:
        sentences = batch_iterator(FLAGS, file)

    print('Loading word2index')
    w2i = pickle.load(
                open(os.path.join(FLAGS.out_dir,'w2i'),'rb')
                )

    print('Opening evaluation file')
    with open(FILE_EVAL,'r') as file:
        sentences_test = batch_iterator(FLAGS, file)

    model = LSTM(FLAGS, train_embeddings = FLAGS.train_embeddings, down_project = FLAGS.down_project)
    model.build_graph()
    model.build_optimizer()

    print('Training LSTM\n')
    loss_history = []
    loss_test_history = []
    perp_test_history = []
    best_loss = np.infty

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        vars_to_save = tf.trainable_variables()
        if model.W_embedding not in vars_to_save: #saver saves only trainable variable by default
            vars_to_save.append(model.W_embedding)
        saver = tf.train.Saver(var_list = vars_to_save, max_to_keep = 2)
        save_directory = os.path.join(FLAGS.out_dir,FLAGS.experiment)

        #deletes content of the directory for saving the trained weights
        if os.path.exists(save_directory):
            shutil.rmtree(save_directory)
            os.mkdir(save_directory)
        else:
            os.mkdir(save_directory)

        sess.run(tf.global_variables_initializer())

        if FLAGS.train_embeddings == False:
            load_embedding(sess, w2i, model.W_embedding, path = os.path.join(FLAGS.data_dir,'wordembeddings-dim100.word2vec'),
                            dim_embedding = FLAGS.embed_size, vocab_size = FLAGS.vocab_size)


        start = time.time()
        while sentences.iteration < FLAGS.num_iterations:
            input = np.stack([word2index(preprocess(sentence, max_length = FLAGS.sentence_length), w2i) for sentence in next(sentences)])

            loss = model.training_step(sess, input, input)

            if sentences.iteration % 100 == 0:
                print('Iteration: {}, training loss: {:.2f}'.format(sentences.iteration, loss))
                loss_history.append(loss)


            if sentences.iteration % 1000 == 0:

                test_iteration_old = sentences_test.iteration

                loss_test = []
                perp_test = []
                while sentences_test.iteration < test_iteration_old + 100:

                    input_test = np.stack([word2index(preprocess(sentence, max_length = FLAGS.sentence_length), w2i) for sentence in next(sentences_test)])
                    loss, cross_entropy = model.evaluate(sess, input = input_test)
                    perp_test.append(perplexity(cross_entropy, input_test))
                    loss_test.append(loss)

                perp_test = sum(perp_test, [])
                perp_test_history.append((np.mean(perp_test),np.median(perp_test)))
                loss_test_history.append(np.mean(loss_test))

                print('\nAverage loss on the testing set: {:.2f}'.format(loss_test_history[-1]))
                print('Average perplexity on 100 batches from the evaluation set: {:.2f}'.format(perp_test_history[-1][0]))
                print('Median perplexity on 100 batches from the evaluation set: {:.2f}'.format(perp_test_history[-1][1]))

                if loss_test_history[-1] < best_loss:
                    best_loss = loss_test_history[-1]
                    saver.save(sess, os.path.join(save_directory,'iteration_{}'.format(sentences.iteration)))

                pickle.dump([loss_history,loss_test_history,perp_test_history],open(os.path.join(save_directory,'losses'),'wb'))

                end = time.time()
                print('Time elapsed: ' + str(end - start) + '\n')
                start = time.time()
