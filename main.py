import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer import train, evaluate, continuation
from utils import get_w2i

FILE_TRAIN = 'data/sentences.train'
FILE_TEST = 'data/sentences_test.txt'

def get_args():
    parser = argparse.ArgumentParser()

    #options
    parser.add_argument('-t', '--train', action='store_true', help='Train the model')
    parser.add_argument('-e', '--evaluate', action='store_true', help='Make evaluation')
    parser.add_argument('-w', '--word2index', action='store_true', help='generates the word2index dictionary')
    parser.add_argument('-emb', '--train_embeddings', action='store_true', help='trains the embedding layer')
    parser.add_argument('-proj', '--down_project', action='store_true', help='adds the projection layer')
    parser.add_argument('-c', '--continuation', action='store_true', help='continuation of sentences')

    #directories
    parser.add_argument('--experiment', type=str, default='A', help='experiment A, B or C')
    parser.add_argument('--out_dir', type=str, default='out', help='output directory')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')

    #model parameters
    parser.add_argument('--vocab_size', type=int, default=20000, help='vocabulary size')
    parser.add_argument('--embed_size', type=int, default=100, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size')
    parser.add_argument('--projection_size', type=int, default=512, help='size of the output of down-projection layer')
    parser.add_argument('--sentence_length', type=int, default=30, help='length of the sentence')
    parser.add_argument('--sentence_cont_length', type=int, default=20, help='length of the continued sentence')
    parser.add_argument('--clip_norm', type=float, default=5.0, help='clipping norm for gradients')

    #training
    parser.add_argument('--batch_size', type=float, default=64, help='batch size')
    parser.add_argument('--num_iterations', type=int, default=30000, help='number of iterations')

    return parser.parse_args()

if __name__ == "__main__":
    FLAGS = get_args()

    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)

    if FLAGS.word2index:
        get_w2i(FLAGS.vocab_size, open(FILE_TRAIN,'r'), FLAGS.out_dir)

    if FLAGS.evaluate:
        evaluate(FLAGS)

    if FLAGS.train:
        train(FLAGS)

    if FLAGS.continuation:
        continuation(FLAGS)
