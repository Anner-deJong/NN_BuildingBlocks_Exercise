################################## Neural Network Model Implementations ##################################

import copy
import numpy as np
from nn_utils import *

# Drop_BN_NN Conv_NN
class Vanilla_NN(object):

    """
    Implements a vanilla neural network object.
    Includes sigmoid activation functions and softmax + cross entropy for final layer

    arguments:
    input_size       = number of features per sample in the training data
    layer_sizes      = list or numpy array, with each entry i corresponding to the amount of hidden neurons in the ith hidden layer
    output_size      = number of output classes for prediction

    functions:
    forward_pass()
    backward_pass()
    test()
    train()
    """
    
    def __init__(self, input_size, layer_sizes, output_size):

        # store the __init__ variables for other functions
        self.D        = input_size
        self.lay_size = layer_sizes
        self.C        = output_size
        
        # create layer variable weights and biases in a dictionary
        self.var = {}
        prev_size = np.hstack((self.D, layer_sizes))                              # temporary lookup array for weight variable sizes
        
        for i, size in enumerate(layer_sizes):                                    # iterate over each layer
            
            cur_W           = ('W%d' % (i+1))
            self.var[cur_W] = np.random.normal(0.0, 0.1, (prev_size[i], size))    # add weight variable
            cur_b           = ('b%d' % (i+1))
            self.var[cur_b] = np.zeros(size)                                      # add bias variable
        
        
        self.var['Wout'] = np.random.normal(0.0, 0.1, (layer_sizes[-1], self.C))  # add output layer weight variable
        self.var['bout'] = np.zeros(self.C)                                       # add output layer bias variable

    
    def forward_pass(self, X, y):
        """
        A single forward pass, affine layers using the sigmoid nonlinearity activation function
        Last layer is implements an affine layer plus a softmax
        Also calculates the loss, prediction, and prediction derivative
        
        arguments:
        X       = numpy matrix, minibatch of N samples, each with self.D features
        y       = numpy matrix, minibatch of one hot encoded labels for same N samples as X
        
        returns:
        loss    = scalar,       softmax prediction loss for current X and y, does NOT contain regularisation loss
        pred    = numpy array,  predicted class for each of N samples
        acc     = scalar,       percentage of correctly predicted classes for current X and y
        dpred   = numpy_array,  derivative of the prediction, (including backprop through softmax layer), internal use for backprop
        cache   = dictionary,   keeps track of all the cahce necessary for backpropagation later on
        """
        
        # create cache dictionary to keep around data required for backward pass
        cache = {}
        
        
        # hidden layer calculations
        inp = X                                                                    # initialise input for following layer iteration 
        for i, size in enumerate(self.lay_size):                                   # iterate over each layer
            
            cur_W                  = self.var[('W%d' % (i+1))]
            cur_b                  = self.var[('b%d' % (i+1))]
            h, cur_cache           = sigmoid_layer_forward(inp, cur_W, cur_b)      # forward layer calculation 
            inp                    = h
            cache[('c%d' % (i+1))] = cur_cache
        
        
        # output layer calculation
        pred_sof, cout = softmax_layer_forward(h, self.var['Wout'], self.var['bout']) # final forward layer calculation
        cache['cout']  = cout
        pred           = np.argmax(pred_sof, axis=1)                               # prediction from softmax distribution to specific class
        acc            = np.sum(pred == np.argmax(y, axis=1)) * 1.0 / X.shape[0]   # final forward layer calculation
        
        
        # loss & prediction derivative calculation
        loss, dpred                = cross_entropy_loss(pred_sof, y)
        
        
        return loss, pred, acc, dpred, cache
    
    
    def backward_pass(self, dpred, cache):
        """
        implements backpropagation through all the layers in the network, saves the derivatives (does NOT update variables!)
        
        arguments:
        dpred   = numpy matrix, contains the derivatives for the output of the final layer
        cache   = dictionary,   forward pass cache necessary for backward pass derivative calculations
        
        returns:
        none
        """
        
        # calculate layer weights and bias derivatives in a dictionary
        self.der = {}
        
        # output layer calculation
        dh, dW, db       = affine_layer_backward(dpred, cache['cout'])             # backpropagate through final layer without softmax
        dout             = dh                                                      # initialise output deriv for following layer iteration
        self.der['Wout'] = dW                                                      # update output layer weight derivatives
        self.der['bout'] = db                                                      # update output layer bias derivatives
        
        # hidden layer calculations
        for i_unrev, size in enumerate(reversed(self.lay_size)):                   # iterate backwards over each layer
            
            i                       = len(self.lay_size)-i_unrev                   # create reversed i. +1 not necessary due to len()
            cur_cac                 = cache[('c%d' % (i))]
            dh, dW, db              = sigmoid_layer_backward(dout, cur_cac)        # backpropagate through layer
            dout                    = dh
            self.der[('W%d' % (i))] = dW                                           # update output layer bias derivatives
            self.der[('b%d' % (i))] = db                                           # update output layer bias derivatives
        
    
    def test(self, X, y):
        """
        implements a forward pass and returns the prediction for testing
        
        arguments:
        X, y    = like in forward_pass()
        
        returns:
        loss    = like in forward_pass()
        pred    =   ,,
        acc     =   ,,
        """
        
        loss, pred, acc, _, _ = self.forward_pass(X, y)
        
        return loss, pred, acc
    
    
    def train(self, X_tr, y_tr, X_va, y_va, number_epochs, batch_size, learning_rate, reg_strength, print_every, verbose):
        """
        this is the core function, it actually trains our model. It keeps track of the best performance on the validation set. 
        Final variables are taken from this best performance 
        
        arguments:
        X_tr, X_va = numpy matrix, minibatch of N samples, each with self.D features
        y_tr, y_va = 
        number_epochs = int,    total number of epochs for training
        batch_size    = int,    batch size of training samples per forward and backward pass
        learning_rate = scalar, learning rate for stochastic gradient descent
        reg_strength  = scalar, regularization strenght (L2 implemented)
        print_every   = int,    print performance during training every this amount of iterations
        verbose       = bool,   print performance during training yes or no
        
        returns:
        tr_loss_hist  = list,   containing training set loss history (without regularisation loss!)
        va_loss_hist  = list,   containing validation set loss history (without regularisation loss!)
        tr_acc_hist   = list,   containing training set accuracy history
        va_acc_hist   = list,   containing validation set accuracy history
        """
        
        # keep the best performing variables around during training:
        best_va_var  = {}
        best_va_acc  = 0
        
        # initialise lists to keep track of the training history
        tr_acc_hist  = []
        va_acc_hist  = []
        tr_loss_hist = []
        va_loss_hist = []
        
        # some dependent parameters
        N              = X_tr.shape[0]
        iterations     = N * number_epochs / batch_size
        iter_per_epoch = iterations / number_epochs

        X_ba, y_ba = get_random_batch(X_tr, y_tr, batch_size)

        for e in range(number_epochs):          # iterate over epochs

            for i in range(int(iter_per_epoch)):     # iterate over iterations per epochs
                
                # get current validation performance
                va_loss, _, va_acc  = self.test(X_va, y_va)
                
                # forward and backpropagate
                #X_ba, y_ba = get_random_batch(X_tr, y_tr, batch_size)
                tr_loss, pred, tr_acc, dpred, cache = self.forward_pass(X_ba, y_ba)
                self.backward_pass(dpred, cache)
                
                # update loss histories
                tr_acc_hist.append(tr_acc)
                va_acc_hist.append(va_acc)
                tr_loss_hist.append(tr_loss)
                va_loss_hist.append(va_loss)
                
                # update the best validation set performing variables
                if va_acc > best_va_acc:
                    best_va_acc = copy.deepcopy(va_acc)
                    best_va_var = copy.deepcopy(self.var)  
                
                # update variables
                for var_key, var  in self.var.items():
                    if 'W' in var_key:                                     # only regularise W matrices, not the bias arrays
                        self.var[var_key] -= reg_strength  * var           # regularisation
                    self.var[var_key] -= learning_rate * self.der[var_key] # derivatives
                
                # print output during training
                if (e*iter_per_epoch + i) % print_every == 0 and verbose:                               
                    print('iteration %4d/%4d, training accuracy: %.2f, training loss: %.4f' \
                          % (e*iter_per_epoch + i, iterations, tr_acc, tr_loss))
            
            # print output during training per epoch
            #if verbose:
            #    print('epoch %2d/%2d, validation accuracy: %.2f'% (e+1, number_epochs, va_acc))
                
        # append final accuracies to history
        _, _, tr_acc = self.test(X_tr, y_tr)
        _, _, va_acc = self.test(X_va, y_va)
        tr_acc_hist.append(tr_acc)
        va_acc_hist.append(va_acc)
        
        # final check for the best validation set performing variables
        if va_acc > best_va_acc:
            best_va_acc = copy.deepcopy(va_acc)
            best_va_var = copy.deepcopy(self.var)  
        
        # replace variables with the best performing ones
        self.var = {}
        self.var = best_va_var
        
        return tr_loss_hist, va_loss_hist, tr_acc_hist, va_acc_hist

class BN_Drop_NN(object):
    """
    Implements almost exactly the same net as Vanilla_NN()
    The only differences are an implementation of dropout and Batch Normalisation in every forward and backward layer

    arguments:
    input_size       = number of features per sample in the training data
    layer_sizes      = list or numpy array, with each entry i corresponding to the amount of hidden neurons in the ith hidden layer
    output_size      = number of output classes for prediction

    functions:
    forward_pass()
    backward_pass()
    test()
    train()
    """

    def __init__(self, input_size, layer_sizes, output_size):


        # store the __init__ variables for other functions
        self.D        = input_size
        self.lay_size = layer_sizes
        self.C        = output_size

        # create layer variable weights and biases in a dictionary
        self.var = {}
        prev_size = np.hstack((self.D, layer_sizes))                              # temporary lookup array for weight variable sizes

        for i, size in enumerate(layer_sizes):                                    # iterate over each layer

            cur_W             = ('W%d' % (i+1))
            self.var[cur_W]   = np.random.normal(0.0, 0.1, (prev_size[i], size))  # add weight variable
            cur_b             = ('b%d' % (i+1))
            self.var[cur_b]   = np.zeros(size)                                    # add bias variable
            cur_gam           = ('gam%d' % (i+1))
            self.var[cur_gam] = np.zeros(size)                                    # add gamma variable
            cur_bet           = ('bet%d' % (i+1))
            self.var[cur_bet] = np.zeros(size)                                    # add beta variable

        self.var['Wout'] = np.random.normal(0.0, 0.1, (layer_sizes[-1], self.C))  # add output layer weight variable
        self.var['bout'] = np.zeros(self.C)                                       # add output layer bias variable

        # keep track of bn_params
        self.bn_params = []
        self.bn_params = [{'mode': 'train'} for i in range(len(layer_sizes))]

    def forward_pass(self, X, y, drop_prob):
        """
        A single forward pass, affine layers using the sigmoid nonlinearity activation function
        Performs dropout and Batch Normalization after affine and before activation
        Last layer is implements an affine layer plus a softmax
        Also calculates the loss, prediction, and prediction derivative

        arguments:
        X       = numpy matrix, minibatch of N samples, each with self.D features
        y       = numpy matrix, minibatch of one hot encoded labels for same N samples as X

        returns:
        loss    = scalar,       softmax prediction loss for current X and y, does NOT contain regularisation loss
        pred    = numpy array,  predicted class for each of N samples
        acc     = scalar,       percentage of correctly predicted classes for current X and y
        dpred   = numpy_array,  derivative of the prediction, (including backprop through softmax layer), internal use for backprop
        cache   = dictionary,   keeps track of all the cahce necessary for backpropagation later on
        """

        # create cache dictionary to keep around data required for backward pass
        cache = {}

        # hidden layer calculations
        inp = X                                                                    # initialise input for following layer iteration
        for i, size in enumerate(self.lay_size):                                   # iterate over each layer

            cur_W                  = self.var[('W%d' % (i+1))]
            cur_b                  = self.var[('b%d' % (i+1))]
            cur_gam                = self.var[('gam%d' % (i+1))]
            cur_bet                = self.var[('bet%d' % (i+1))]
            h, cur_cache           = BN_Dr_sig_layer_forward(inp, cur_W, cur_b,    # forward layer calculation
                                                             drop_prob,            # dropout parameters
                                                             cur_gam, cur_bet, self.bn_params[i])   # batch normalization parameters
            inp                    = h
            cache[('c%d' % (i+1))] = cur_cache


        # output layer calculation
        pred_sof, cout = softmax_layer_forward(h, self.var['Wout'], self.var['bout']) # final forward layer calculation
        cache['cout']  = cout
        pred           = np.argmax(pred_sof, axis=1)                               # prediction from softmax distribution to specific class
        acc            = np.sum(pred == np.argmax(y, axis=1)) * 1.0 / X.shape[0]   # final forward layer calculation


        # loss & prediction derivative calculation
        loss, dpred                = cross_entropy_loss(pred_sof, y)


        return loss, pred, acc, dpred, cache


    def backward_pass(self, dpred, cache):
        """
        implements backpropagation through all the layers in the network, saves the derivatives (does NOT update variables!)

        arguments:
        dpred   = numpy matrix, contains the derivatives for the output of the final layer
        cache   = dictionary,   forward pass cache necessary for backward pass derivative calculations

        returns:
        none
        """

        # calculate layer weights and bias derivatives in a dictionary
        self.der = {}

        # output layer calculation
        dh, dW, db       = affine_layer_backward(dpred, cache['cout'])             # backpropagate through final layer without softmax
        dout             = dh                                                      # initialise output deriv for following layer iteration
        self.der['Wout'] = dW                                                      # update output layer weight derivatives
        self.der['bout'] = db                                                      # update output layer bias derivatives

        # hidden layer calculations
        for i_unrev, size in enumerate(reversed(self.lay_size)):                   # iterate backwards over each layer

            i                         = len(self.lay_size)-i_unrev                 # create reversed i. +1 not necessary due to len()
            cur_cac                   = cache[('c%d' % (i))]
            dh, dW, db, dgam, dbet    = BN_Dr_sig_layer_backward(dout, cur_cac)    # backpropagate through layer
            dout                      = dh
            self.der[('W%d' % (i))]   = dW                                         # update weight derivatives for layer i
            self.der[('b%d' % (i))]   = db                                         # update bias derivatives layer i
            self.der[('gam%d' % (i))] = dgam                                       # update gamma (BN) derivatives layer i
            self.der[('bet%d' % (i))] = dbet                                       # update beta (BN) derivatives layer i


    def test(self, X, y):
        """
        implements a forward pass and returns the prediction for testing

        arguments:
        X, y    = like in forward_pass()

        returns:
        loss    = like in forward_pass()
        pred    =   ,,
        acc     =   ,,
        """

        loss, pred, acc, _, _ = self.forward_pass(X, y, drop_prob=0.0)

        return loss, pred, acc


    def train(self, X_tr, y_tr, X_va, y_va, number_epochs, batch_size, learning_rate, reg_strength, drop_prob, print_every, verbose):
        """
        this is the core function, it actually trains our model. It keeps track of the best performance on the validation set.
        Final variables are taken from this best performance

        arguments:
        X_tr, X_va = numpy matrix, minibatch of N samples, each with self.D features
        y_tr, y_va =
        number_epochs = int,    total number of epochs for training
        batch_size    = int,    batch size of training samples per forward and backward pass
        learning_rate = scalar, learning rate for stochastic gradient descent
        reg_strength  = scalar, regularization strenght (L2 implemented)
        print_every   = int,    print performance during training every this amount of iterations
        verbose       = bool,   print performance during training yes or no

        returns:
        tr_loss_hist  = list,   containing training set loss history (without regularisation loss!)
        va_loss_hist  = list,   containing validation set loss history (without regularisation loss!)
        tr_acc_hist   = list,   containing training set accuracy history
        va_acc_hist   = list,   containing validation set accuracy history
        """

        # keep the best performing variables around during training:
        best_va_var  = {}
        best_va_acc  = 0

        # initialise lists to keep track of the training history
        tr_acc_hist  = []
        va_acc_hist  = []
        tr_loss_hist = []
        va_loss_hist = []

        # some dependent parameters
        N              = X_tr.shape[0]
        iterations     = N * number_epochs / batch_size
        iter_per_epoch = iterations / number_epochs

        for e in range(number_epochs):          # iterate over epochs

            for i in range(int(iter_per_epoch)):     # iterate over iterations per epochs

                # get current validation performance
                va_loss, _, va_acc  = self.test(X_va, y_va)

                # forward and backpropagate
                X_ba, y_ba = get_random_batch(X_tr, y_tr, batch_size)
                tr_loss, pred, tr_acc, dpred, cache = self.forward_pass(X_ba, y_ba, drop_prob)
                self.backward_pass(dpred, cache)

                # update loss histories
                tr_acc_hist.append(tr_acc)
                va_acc_hist.append(va_acc)
                tr_loss_hist.append(tr_loss)
                va_loss_hist.append(va_loss)

                # update the best validation set performing variables
                if va_acc > best_va_acc:
                    best_va_acc = copy.deepcopy(va_acc)
                    best_va_var = copy.deepcopy(self.var)

                # update variables
                for var_key, var  in self.var.items():
                    if 'W' in var_key:                                     # only regularise W matrices, not the bias arrays
                        self.var[var_key] -= reg_strength  * var           # regularisation
                    self.var[var_key] -= learning_rate * self.der[var_key] # derivatives

                # print output during training
                if (e*iter_per_epoch + i) % print_every == 0 and verbose:
                    print('iteration %4d/%4d, training accuracy: %.2f, training loss: %.4f' \
                          % (e*iter_per_epoch + i, iterations, tr_acc, tr_loss))

            # print output during training per epoch
            #if verbose:
            #    print('epoch %2d/%2d, validation accuracy: %.2f'% (e+1, number_epochs, va_acc))

        # append final accuracies to history
        _, _, tr_acc = self.test(X_tr, y_tr)
        _, _, va_acc = self.test(X_va, y_va)
        tr_acc_hist.append(tr_acc)
        va_acc_hist.append(va_acc)

        # final check for the best validation set performing variables
        if va_acc > best_va_acc:
            best_va_acc = copy.deepcopy(va_acc)
            best_va_var = copy.deepcopy(self.var)

        # replace variables with the best performing ones
        self.var = {}
        self.var = best_va_var

        return tr_loss_hist, va_loss_hist, tr_acc_hist, va_acc_hist
    
class Conv_NN(object):
    """
    Implements a convolutional neural network with first conv layers followed by some fully connected layers
    Furthermore includes Dropout, batch normalization, sigmoid activation functions, and softmax + cross entropy for final layer
        Dropout and batch normalization is only applied to the fully connected layers
        However, applying batch norm alization to conv layers might be useful
        Applying dropout to conv layers is not very commen and not recommended
            (co-adaptation isn't mitigated when weights are only spatially put to 0,
            and when a weight mask has weights at zero anywhere at its input this is rather strong)

    arguments:
    input_size       = number of features per sample in the training data
    layer_sizes      = list or numpy array, with each entry i corresponding to the amount of hidden neurons in the ith hidden layer
    output_size      = number of output classes for prediction

    functions:
    forward_pass()
    backward_pass()
    test()
    train()
    """

    def __init__(self, input_dim, conv_layer_sizes, conv_window_size, affine_layer_sizes, output_size):


        # store the __init__ variables for other functions
        self.H, self.W, self.color = input_dim
        self.conv_W    = conv_window_size
        self.conv_size = conv_layer_sizes
        self.fc_size   = affine_layer_sizes
        self.C         = output_size

        # make sure conv weight window size is 3, 5 or 7:
        # (this does not have to be strict, feel free to experiment with other sizes)
        if self.conv_W != 3 and self.conv_W != 5 and self.conv_W != 7:
            raise AssertionError('convolutional weight window size is not 3, 5 or 7, but {0}'.format(self.conv_W))

        # create convolutional layer variable weights and biases in a dictionary
        self.var  = {}
        prev_size = np.hstack((self.color, conv_layer_sizes))  # temporary lookup array for weight variable sizes
        weight_sc = 1e-3  # weight scale for conv layer initialisation
        for i, size in enumerate(conv_layer_sizes):  # iterate over each layer

            cur_W = ('W%d' % (i + 1))        # add weight variable
            self.var[cur_W] = np.random.randn(size, prev_size[i],  self.conv_W, self.conv_W ) * weight_sc
            cur_b = ('b%d' % (i + 1))
            self.var[cur_b] = np.zeros(size)    # add bias variable

        # create fully connected layer variable weights and biases in a dictionary
        prev_size = np.hstack(((self.H * self.W * self.conv_size[-1]), affine_layer_sizes)) # temporary lookup array for weight variable sizes

        for i, size in enumerate(affine_layer_sizes):                             # iterate over each layer
            e = len(conv_layer_sizes) + i                                         # update layer number after conv layers

            cur_W             = ('W%d' % (e+1))
            self.var[cur_W]   = np.random.normal(0.0, 0.1, (prev_size[i], size))  # add weight variable
            cur_b             = ('b%d' % (e+1))
            self.var[cur_b]   = np.zeros(size)                                    # add bias variable
            cur_gam           = ('gam%d' % (e+1))
            self.var[cur_gam] = np.zeros(size)                                    # add gamma variable
            cur_bet           = ('bet%d' % (e+1))
            self.var[cur_bet] = np.zeros(size)                                    # add beta variable

        self.var['Wout'] = np.random.normal(0.0, 0.1, (affine_layer_sizes[-1], self.C))  # add output layer weight variable
        self.var['bout'] = np.zeros(self.C)                                       # add output layer bias variable

        # keep track of bn_params
        self.bn_params = []
        self.bn_params = [{'mode': 'train'} for i in range(len(affine_layer_sizes))]

    def forward_pass(self, X, y, drop_prob):
        """
        A single forward pass, affine layers using the sigmoid nonlinearity activation function
        Performs dropout and Batch Normalization after affine and before activation
        Last layer is implements an affine layer plus a softmax
        Also calculates the loss, prediction, and prediction derivative

        arguments:
        X       = numpy matrix, minibatch of N samples, each with self.D features
        y       = numpy matrix, minibatch of one hot encoded labels for same N samples as X

        returns:
        loss    = scalar,       softmax prediction loss for current X and y, does NOT contain regularisation loss
        pred    = numpy array,  predicted class for each of N samples
        acc     = scalar,       percentage of correctly predicted classes for current X and y
        dpred   = numpy_array,  derivative of the prediction, (including backprop through softmax layer), internal use for backprop
        cache   = dictionary,   keeps track of all the cahce necessary for backpropagation later on
        """

        # keep track of amount of batches
        N = X.shape[0]

        # create cache dictionary to keep around data required for backward pass
        cache = {}

        # convolutional layer calculations
        inp = X                                                                    # initialise input for following layer iteration
        for i, size in enumerate(self.conv_size):                                  # iterate over each layer
            cur_W                  = self.var[('W%d' % (i+1))]
            cur_b                  = self.var[('b%d' % (i+1))]
            h, cur_cache           = sig_conv_layer_forward(inp, cur_W, cur_b)    # forward layer calculation

            inp                    = h
            cache[('c%d' % (i+1))] = cur_cache

        # flatten
        self.unflat_shape = inp.shape                                              # remember the shape for unflattening in the backward pass
        flat = inp.reshape((N, -1))

        # fully connected layer calculations
        inp = flat                                                                 # initialise input for following layer iteration
        for i, size in enumerate(self.fc_size):                                    # iterate over each layer
            e = len(self.conv_size) + i
            cur_W                  = self.var[('W%d' % (e+1))]
            cur_b                  = self.var[('b%d' % (e+1))]
            cur_gam                = self.var[('gam%d' % (e+1))]
            cur_bet                = self.var[('bet%d' % (e+1))]
            h, cur_cache           = BN_Dr_sig_layer_forward(inp, cur_W, cur_b,    # forward layer calculation
                                                             drop_prob,            # dropout parameters
                                                             cur_gam, cur_bet, self.bn_params[i])   # batch normalization parameters
            inp                    = h
            cache[('c%d' % (e+1))] = cur_cache


        # output layer calculation
        pred_sof, cout = softmax_layer_forward(h, self.var['Wout'], self.var['bout']) # final forward layer calculation
        cache['cout']  = cout
        pred           = np.argmax(pred_sof, axis=1)                               # prediction from softmax distribution to specific class
        acc            = np.sum(pred == np.argmax(y, axis=1)) * 1.0 / X.shape[0]   # final forward layer calculation


        # loss & prediction derivative calculation
        loss, dpred                = cross_entropy_loss(pred_sof, y)


        return loss, pred, acc, dpred, cache

    def backward_pass(self, dpred, cache):
        """
        implements backpropagation through all the layers in the network, saves the derivatives (does NOT update variables!)

        arguments:
        dpred   = numpy matrix, contains the derivatives for the output of the final layer
        cache   = dictionary,   forward pass cache necessary for backward pass derivative calculations

        returns:
        none
        """

        # calculate layer weights and bias derivatives in a dictionary
        self.der = {}

        # output layer calculation
        dh, dW, db       = affine_layer_backward(dpred, cache['cout'])             # backpropagate through final layer without softmax
        dout             = dh                                                      # initialise output deriv for following layer iteration
        self.der['Wout'] = dW                                                      # update output layer weight derivatives
        self.der['bout'] = db                                                      # update output layer bias derivatives

        # fully connected layer calculations
        for i_unrev, size in enumerate(reversed(self.fc_size)):                    # iterate backwards over each layer

            e                         = len(self.fc_size+self.conv_size)-i_unrev   # create reversed i. +1 not necessary due to len()
            cur_cac                   = cache[('c%d' % (e))]
            dh, dW, db, dgam, dbet    = BN_Dr_sig_layer_backward(dout, cur_cac)    # backpropagate through layer
            dout                      = dh
            self.der[('W%d' % (e))]   = dW                                         # update weight derivatives for layer i
            self.der[('b%d' % (e))]   = db                                         # update bias derivatives layer i
            self.der[('gam%d' % (e))] = dgam                                       # update gamma (BN) derivatives layer i
            self.der[('bet%d' % (e))] = dbet                                       # update beta (BN) derivatives layer i

        # unflatten
        dout = dh.reshape(self.unflat_shape)

        # convolutional layer calculations
        for i_unrev, size in enumerate(reversed(self.conv_size)):                  # iterate backwards over each layer

            i                         = len(self.conv_size)-i_unrev                # create reversed i. +1 not necessary due to len()
            cur_cac                   = cache[('c%d' % (i))]
            dh, dW, db                = sig_conv_layer_backward(dout, cur_cac)     # backpropagate through layer
            dout                      = dh
            self.der[('W%d' % (i))]   = dW                                         # update weight derivatives for layer i
            self.der[('b%d' % (i))]   = db                                         # update bias derivatives layer i

    def test(self, X, y):
        """
        implements a forward pass and returns the prediction for testing

        arguments:
        X, y    = like in forward_pass()

        returns:
        loss    = like in forward_pass()
        pred    =   ,,
        acc     =   ,,
        """

        loss, pred, acc, _, _ = self.forward_pass(X, y, drop_prob=0.0)

        return loss, pred, acc

    def train(self, X_tr, y_tr, X_va, y_va, number_epochs, batch_size, learning_rate, reg_strength, drop_prob, print_every, verbose):
        """
        this is the core function, it actually trains our model. It keeps track of the best performance on the validation set.
        Final variables are taken from this best performance

        arguments:
        X_tr, X_va = numpy matrix, minibatch of N samples, each with self.D features
        y_tr, y_va =
        number_epochs = int,    total number of epochs for training
        batch_size    = int,    batch size of training samples per forward and backward pass
        learning_rate = scalar, learning rate for stochastic gradient descent
        reg_strength  = scalar, regularization strenght (L2 implemented)
        print_every   = int,    print performance during training every this amount of iterations
        verbose       = bool,   print performance during training yes or no

        returns:
        tr_loss_hist  = list,   containing training set loss history (without regularisation loss!)
        va_loss_hist  = list,   containing validation set loss history (without regularisation loss!)
        tr_acc_hist   = list,   containing training set accuracy history
        va_acc_hist   = list,   containing validation set accuracy history
        """

        # keep the best performing variables around during training:
        best_va_var  = {}
        best_va_acc  = 0

        # initialise lists to keep track of the training history
        tr_acc_hist  = []
        va_acc_hist  = []
        tr_loss_hist = []
        va_loss_hist = []

        # some dependent parameters
        N              = X_tr.shape[0]
        iterations     = N * number_epochs / batch_size
        iter_per_epoch = iterations / number_epochs

        for e in range(number_epochs):          # iterate over epochs

            for i in range(int(iter_per_epoch)):     # iterate over iterations per epochs

                # get current validation performance
                va_loss, _, va_acc  = self.test(X_va, y_va)

                # forward and backpropagate
                X_ba, y_ba = get_random_batch(X_tr, y_tr, batch_size)
                tr_loss, pred, tr_acc, dpred, cache = self.forward_pass(X_ba, y_ba, drop_prob)
                self.backward_pass(dpred, cache)

                # update loss histories
                tr_acc_hist.append(tr_acc)
                va_acc_hist.append(va_acc)
                tr_loss_hist.append(tr_loss)
                va_loss_hist.append(va_loss)

                # update the best validation set performing variables
                if va_acc > best_va_acc:
                    best_va_acc = copy.deepcopy(va_acc)
                    best_va_var = copy.deepcopy(self.var)

                # update variables
                for var_key, var  in self.var.items():
                    if 'W' in var_key:                                     # only regularise W matrices, not the bias arrays
                        self.var[var_key] -= reg_strength  * var           # regularisation
                    self.var[var_key] -= learning_rate * self.der[var_key] # derivatives

                # print output during training
                if (e*iter_per_epoch + i) % print_every == 0 and verbose:
                    print('iteration %4d/%4d, training accuracy: %.2f, training loss: %.4f' \
                          % (e*iter_per_epoch + i, iterations, tr_acc, tr_loss))

            # print output during training per epoch
            #if verbose:
            #    print('epoch %2d/%2d, validation accuracy: %.2f'% (e+1, number_epochs, va_acc))

        # append final accuracies to history
        _, _, tr_acc = self.test(X_tr, y_tr)
        _, _, va_acc = self.test(X_va, y_va)
        tr_acc_hist.append(tr_acc)
        va_acc_hist.append(va_acc)

        # final check for the best validation set performing variables
        if va_acc > best_va_acc:
            best_va_acc = copy.deepcopy(va_acc)
            best_va_var = copy.deepcopy(self.var)

        # replace variables with the best performing ones
        self.var = {}
        self.var = best_va_var

        return tr_loss_hist, va_loss_hist, tr_acc_hist, va_acc_hist

    
    
    
        