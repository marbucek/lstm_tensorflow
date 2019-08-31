import numpy as np
import pickle
import os
import time
from tensorflow.keras.utils import to_categorical
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_w2i(vocab_size, training_file, out_dir, save = True):
    '''
    vocab_size:     size of vocabulary
    training_file:  path to training file
    out_dir:        directory to save the word2index dictionary
    '''

    #assign <bos>, <eos>, <unk> and <pad> to the first four indices
    w2i = ['<bos>','<eos>','<unk>','<pad>']

    print('Reading training file and counting words...')
    data = pd.read_csv(training_file)
    counts = {}
    for _, row in data.iterrows():

        tokens = row.split() #creates list of tokens present at current row

        for token in tokens:
            if token in counts.keys():
                counts[token] += 1
            else:
                counts[token] = 1

    print('Restricting word2index to the vocabulary size...')
    #restrict items in the w2i to the vocabulary size
    all_tokens = []; num_counts = []
    for key, val in counts.items():
        all_tokens.append(key)
        num_counts.append(val)

    sorted_tokens = [token for _, token in sorted(zip(num_counts, all_tokens), reverse = True)]

    for id, token in enumerate(sorted_tokens[0:vocab_size - len(w2i_default)]):
        w2i.append(token)

    print(f'Saving the word2index list to {out_dir}/w2i')
    pickle.dump(w2i, open(os.path.join(out_dir,'w2i'),'wb'))
    return w2i

# def get_w2i(vocab_size, training_file, out_dir):
#
#     #assign <bos>, <eos>, <unk> and <pad> to the first four indices
#     w2i = np.array(['<bos>','<eos>','<unk>','<pad>'])
#
#     print('Reading training file')
#     rows = np.concatenate([row.split() for row in training_file])
#     print('Reading training file completed')
#     words, counts = np.unique([word for word in rows.reshape(-1) if word not in w2i], return_counts = True)
#     order = np.argsort(counts)[::-1]
#     words = words[order[:vocab_size - len(w2i)]]
#
#     w2i = list(np.concatenate((w2i, words), axis = 0))
#
#     pickle.dump(w2i, open(os.path.join(out_dir,'w2i'),'wb'))

def word2index(sentence, w2i):

    indices = np.zeros(len(sentence))

    for id, word in enumerate(sentence):
        try:
            indices[id] = w2i.index(word)
        except: #if unknown word
            indices[id] = 2

    return indices.astype(np.int32)


def index2word(indices, w2i):

    sentence = [w2i[index] for index in indices]

    return sentence

def preprocess(sentence, max_length):

    new_sentence = sentence.copy()
    new_sentence.insert(0,'<bos>')
    new_sentence.append('<eos>')

    for i in range(len(new_sentence), max_length):
        new_sentence.append('<pad>')

    return new_sentence

def perplexity(cross_entropy, ground_truth):

    cross_entropy[ground_truth[:,1:] == 3] = np.nan #ignoring the <pad> terms
    perp = list(np.exp(np.nanmean(cross_entropy, axis = -1))) #np.nanmean ignores the nan values

    return perp
