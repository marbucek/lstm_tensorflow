# Before running the code
Download the preprocessed training file [sentences.train](https://www.icloud.com/iclouddrive/0fZUO7iKl9A5QdQinOLa2voGQ#sentences) and save it to the directory _data/_.

# Create word2index dictionary
This operation should be run once before starting with training. Running

```
python main.py -w
```

will save the word2index dictionary into the folder _out/_.


# Train the network
To train the network including embedding layer for the default of 30000 iterations, batch size 64, hidden state size 512 and vocabulary size 20000 (experiment A), run

```
python main.py -t --train_embeddings --experiment A
```

To evaluate the experiment A and generate the file with perplexities, run

```
python main.py -e --experiment A
```

Experiment B: using pre-trained embeddings. Save the file _wordembeddings-dim100.word2vec_ into the directory _data/_ and run
```
python main.py -t --experiment B
python main.py -e --experiment B
```

Experiment C: include the down-projection layer and change the hidden state size to 1024:

```
python main.py -t -proj --hidden_size 1024 --experiment C
python main.py -e -proj --hidden_size 1024 --experiment C
```

Various other parameters can be adjusted, see the file _main.py_.

To generate the continued sentences, run the code with the ```-c``` optional argument, e.g.
```
python main.py -c -proj --hidden_size 1024 --experiment C
```

This code has been created as part of the coursework for the ETH course on [Natural Language Understanding](http://www.da.inf.ethz.ch/teaching/2019/NLU/).

# Explore the trained model
Alternatively, the training part can be skipped: the trained model from experiment C is saved in the directory _out/C/_. See the file [jupyter notebook](./continuation.ipynb) for examples of completed sentences.
