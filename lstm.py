################################################
# INSTRUCTIONS TO RUN THE CODE:
# - Obtain a .txt file of the book contents.
# - Read the text file and see the unique characters in the file. Then make a list of unwanted characters to be eliminated from the text.
# - Now fix a window length and stride length (of characters) to train the model. Given here are 40 and 5 respectively
# - One-hot bit coding is done for all the unique characters in the text
# - Choose the optimizer for the model in network_build() method
# - Choose the number of layers in the network_build() method. Default set to 1 layer with dropout of 0.1
# - Set the number of epochs as no.of iterations and the different 'diversity' values
# - Decreasing the diversity from 1 to some lower number (e.g. 0.5) makes the RNN more confident, but also more conservative in its samples.
# - Conversely, higher diversityeratures will give more diversity but at cost of more mistakes (e.g. spelling mistakes, etc)
# - Set the number of characters to be predicted in each iteration and for each value of diversity
# - Set the batch-size for training the model
# - Run the script to see the predicted characters for each 'diversity' value after every epoch along with the loss.  
################################################

################################################
# Loading the depenencies
from __future__ import print_function
from keras.models import Sequential # For building the model
from keras.layers import Dense, Activation, Dropout # Layers in the model
from keras.layers import LSTM 
from keras.optimizers import RMSprop # Optimizer
from keras.utils.data_utils import get_file
import numpy as np
import random 
import sys # To save logs
import matplotlib.pyplot as plt # To plot 
np.random.seed(7) # Setting seed for reproducability purpose
print('Seed Set to 7...')
# from keras import backend as K
# K.set_image_dim_ordering('th') # Since I'm using dim of images as channels,width,height
################################################

################################################
def network_build():
    print('Build network...')
    network = Sequential()
    network.add(LSTM(128, input_shape=(maxlen, len(chars))))
    network.add(Dropout(0.10))
    # network.add(LSTM(128))
    # network.add(Dropout(0.25))
    network.add(Dense(len(chars)))
    network.add(Activation('softmax'))
    # network.load_weights("lstm_weights.h5") # To run with already trained weights
    optimizer = RMSprop(lr=0.01) 
    network.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # network.compile(loss='categorical_crossentropy', optimizer=('adadelta'))
    return network

def sample(preds, diversity=1.0):
    # helper function to sample an index from a probability array 
    # diversity = 1 is default value if no diversity is specified
    preds = np.asarray(preds).astype('float32')
    preds = np.log(preds) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)
################################################

######################################################
# train the network, output generated text after each iteration
if __name__ == "__main__":
    text = open('HumanAction.txt').read().lower()
    # Comment next line and uncomment the succeeding line if new-line characters aren't yet removed. Else leave as it is. 
    unwanted=['\n', '\r', '\x80', '\x84', '\x88', '\x89', '\x92', '\x93', '\x94', '\x96', '\x98', '\x99', '\x9c', '\x9d', '\x9e', '\x9f', '\xa0', '\xa1', '\xa3', '\xa4', '\xa6', '\xa7', '\xa8', '\xa9', '\xaa', '\xab', '\xae', '\xaf', '\xb6', '\xb7', '\xb9', '\xbb', '\xbc', '\xbe', '\xc2', '\xc3', '\xc5', '\xc6', '\xce', '\xcf', '\xe2']
    # unwanted=['\x80', '\x84', '\x88', '\x89', '\x92', '\x93', '\x94', '\x96', '\x98', '\x99', '\x9c', '\x9d', '\x9e', '\x9f', '\xa0', '\xa1', '\xa3', '\xa4', '\xa6', '\xa7', '\xa8', '\xa9', '\xaa', '\xab', '\xae', '\xaf', '\xb6', '\xb7', '\xb9', '\xbb', '\xbc', '\xbe', '\xc2', '\xc3', '\xc5', '\xc6', '\xce', '\xcf', '\xe2']
    text=[x for x in text if not x in unwanted]
    text=''.join(text) # From list to string
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40
    step = 5
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # Iterations
    for iteration in range(1, 31):
        print()
        print('-' * 60)
        print('Iteration: ', iteration)
        network = network_build() 
        
        # oldStdout = sys.stdout # Saving the verbose information in  log file (text file)
        # file = open('logFileLSTM2.txt', 'w')
        # sys.stdout = file     # Callback to save weights of the mdel whenever the validation accuracy improves
        history = network.fit(X, y, batch_size=128, nb_epoch=1)
        network.save_weights('lstm_weights2.h5')
        # sys.stdout = oldStdout

        # # Plot of loss
        # plt.plot(history.history['loss'])
        # plt.title('Loss Plot')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.show()

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.35, 0.70]:
        # for diversity in [0.2, 0.5, 1.0]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = network.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()