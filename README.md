**Problem Statement:**

Train an LSTM on Human Action by Ludwig von Mises . This book is
supposedly the best defense of capitalism ever written. Then generate
five samples of random text that sound like his work

**Note:**

Set of instructions to run the code are provided in the main.py file

**Experiments and Analysis:**

This code works with a fair understanding of LSTM and its implementation
using *keras* library with *theano* backend. It is recommended to run
the python script using GPU, as recurrent models are quite
computationally intensive. For this problem, I choose a simple model
with a single LSTM layer with a dropout of 0.1 and a dense-softmax
layer. I have use d the *optimizer* as *RMSprop* with *learning
rate=0.01* and the loss function being *categorical\_crossentropy*.

**Preprocessing:**

Firstly, a *.txt* version of the book was obtained from *.epub* file
available online. After analyzing the contents of the book and observing
random snippets of text, it has been found that pre-processing of the
data is required as it contained unwanted symbols and characters.

After several experiments, it has been found that at least 20 epochs are
required before the generated text starts sounding coherent. To observe
meaningful results, it is advisable to run this script if there are at
least 1M characters in the corpus. One reason for training the model at
character level rather than at word level is that unique characters in
text are finitely small in number compared to the number of unique
words. Since one-hot bit encoding is adopted here, it is a better choice
to train the model at character level.

To train the model, a 40-length character string window was recorded as
a sentence and moving the window with a stride 5, all sentences were
recorded. A total of 0.49M sentences were obtained with the total
character length in the text being close to \~2.4M. The total number of
unique characters was 62.

**Diversity:**

The *diversity* factor of the softmax can be varied from 0 to 1 range
during sampling. Decreasing the *diversity* value from 1 to some lower
number (e.g. 0.2) makes the model more confident, but also more
conservative in its samples. And conversely, higher values of
*diversity* will give more variety but at cost of more mistakes (e.g.
spelling mistakes, etc).

**No. of Epochs:**

The model was trained for 30 epochs with a *batch-size* of 128 and the
model’s *weights* after the 30 epochs were saved in the attached
*lstm\_weights.h5*. After every epoch, given any random 40-characters
length string from the text, the model predicted the next 400 characters
following the given string. The loss during the training phase was also
saved after each epoch and it was observed that it got saturated after
13-15 epochs. A detailed result is available in the *log.txt* file
attached. The results for some of the epochs are given below:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Epoch 1/30

490279/490279 \[==============================\] - 134s - loss: 1.6788

----- diversity: 0.35

----- Generating with seed: "people scorn and disparage capitalism an"

people scorn and disparage capitalism an *d the money of the are not the
wage of the secondard*

*as the mistake conditions of the position of the process of the servine
of the money and a*

*produces and the produce of the prodeces of the social are man consumed
and action of the*

*money of the are an individuals and in the position of the produces and
the such a produced to*

*the social are the prodeces of the compleing the conditions of the
wage*

----- diversity: 0.7

----- Generating with seed: "people scorn and disparage capitalism an"

people scorn and disparage capitalism an *d the economic point of the
fanling of the supply of*

*one postiolitious conditions of the result the are demand goods of
powities to be tender within*

*the goods in the manding money of a more moder that men can be
commodity or with result*

*perfect seller not demandind instince existive business in a prinect
with result destriction of the*

*rest right of a muster measure souction which them and labor to*

Epoch 10/30

490279/490279 \[==============================\] - 132s - loss: 1.6265

----- diversity: 0.35

----- Generating with seed: " which people considereither with regard"

which people considereither with regard *to the power of the domestic
exchange of the market*

*economy. they different and the productivity of money of the
competitive of the state of the*

*results of the realize of the fact the states of the solution of a man
between the reality of the*

*state of the states of the consumer the dividence of the socialist
exchange as a money of the*

*consumption of the productivity of the expended of the more of*

----- diversity: 0.7

----- Generating with seed: " which people considereither with regard"

which people considereither with regard *. they are it is not market
state becaupanting the best of*

*the signial countries of problems are same in their siles power to
animally rule exchange. they*

*work is no period. what is means in the fact the more the seeficed
exchange are emerged to*

*activitimences from them solection, ern and labor so, conditions. they
do not socialism and*

*owners to recommert. it is government readed of the market pro*

Epoch 20/30

490279/490279 \[==============================\] - 133s - loss: 1.6424

----- diversity: 0.35

----- Generating with seed: " time preceding the credit expansion a s"

time preceding the credit expansion a s *ubstitution of the controctory
of the conditions of the*

*conditions of production in the contrort of the restrict do not are
individual are an income and the*

*conditions of production of the conditions of the conditions of the
controrthough of the*

*consument of the contrict in the contrict and international proposing
contrort to the*

*entrepreneure of production of production of the contrort to an incr*

----- diversity: 0.7

----- Generating with seed: " time preceding the credit expansion a s"

time preceding the credit expansion a s *ervices not a thounder who
entirely cannot term without*

*a more offers the fines market personing contropo that concerneration
which interest*

*reliberation and he does not production of the amount of the provide
only the policy on the must*

*recoursory are contrict mather employmoners araination.there is
persoroblech production*

*between withor contriciation is the concept of conditions or the great
to*

Epoch 30/30

490279/490279 \[==============================\] - 135s - loss: 1.6228

----- diversity: 0.35

----- Generating with seed: "s are by the interplay of the forces ope"

s are by the interplay of the forces ope *ration of an important the are
in the prices of the prices of*

*the comporation of the comporitions of man of the process the society
and the action of the*

*prices of the complested the construction of the large of the process
the profitation of the money*

*and the economics and the consumers' that the consumers. but the
comples of the fact that the*

*ends of the society of the monopolistic process of in*

----- diversity: 0.7

----- Generating with seed: "s are by the interplay of the forces ope"

s are by the interplay of the forces ope *raty products which the money
and and of at curred of*

*some the gonding or of bank. if they brus to be suised for a marx. they
are time. but which is the*

*competition and the events of he it that the pricas of the money. for
errors and prices. the prices.*

*and the value that the forcun of the forme hard, they does not resprour
inventating and the are*

*purchasitativility and are from the solling the mo*

In the results above, the predicted characters are in *italics*. As it
can be seen, the

*diversity* factor plays a major role in predicting the output. A lower
value of it will result in

conservative results while a higher value is showing variety in the
results. On the whole,

it is nice to experiment and see the power of LSTMs in the domain of
text prediction.

Acknowledgement:

**Acknowledgements:**

Thanks to Mohan Sai Krishna for lending his laptop with GPU (GeForce GTX
970M)

enabled and cuDNN installed.

**References:**

1.
https://github.com/fchollet/keras/blob/master/examples/lstm\_text\_generation.py

2\. http://karpathy.github.io/2015/05/21/rnn-effectiveness/

3\. http://colah.github.io/posts/2015-08-Understanding-LSTMs/
