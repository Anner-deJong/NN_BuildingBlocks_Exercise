# Welcome!

This ipython notebook implements three simple Neural Nets that try to classify the MNIST dataset:

* Vanilla_NN:
Vanilla Neural Network

* Drop_BN_NN:
Slightly fancier Neural Network with Dropout and Batch Normalization

* Conv_NN:
Simple Convolutional Neural Network

This code is mostly aimed for those who are familiar with the basics of Neural Networks, and want to see and understand an actual practical implementation.

As such, this code doesn't use any higher level libraries such as Tensorflow. (It only makes extensive use of Numpy.)

It also means that it is much easier to follow variables around, and see how different building blocks interact with each other to form a full-fledged net. This all comes at the drawback of not being able to use very optimized implementations.

Therefore, although the presented net seems quite robust, it is also hard to optimize and converge, and have very good performance. Again, this code is meant for understanding and 'playing around', not aimed at high-end performance.



### I hope this gives an easy to follow code and a good overview of what happens inside a Neural Nets step by step, and that many people may benefit from tinkering with it.

## Anner

Files description:

* Data_MNIST:		         
folder that should contain all the data, please download these yourself

* Models.py:             
script that contains athe three networks as python classes

* NN_utils.py:
script that contains all library of main neural network functions

* NeuralNets.ipynb
ipython notebook that instantiates the network classes from Models.py, and trains them etc. More description inside.

* NN_SinglePage.ipynb
Separate, stand alone ipython notebook in case you just want a single file of code. (Requires data separate though!)


Lacking:

* ReLU layer, other activation functions
* Optimized initialisation of the weight variables such as Glorot/Xavier initialisation
* Visualize different nets training characteristics next to each other
* Create parent class, or integrate three classes into on (they are not that different)
* Further checks why there is a lack of convergence
