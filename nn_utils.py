################################## Neural Network Hulp/Utility functions ##################################
# Here you can find a lot of building blocks for Neural Nets, sorry for not providing more detailed explanations
# Some of these were build based on, or even copied from, my implementation for Stanford's cs231n class's homework assignments
import numpy as np

###################### Affine layer ########################

def affine_layer_forward(inp, W, b):
    # implements vanilla foward neural net layer, assumes all input to be numpy
    out   = np.add(np.matmul(inp, W), b)
    cache = (inp, W)
    return out, cache

def affine_layer_backward(dout, cache):
    # implements derivative of vanilla foward neural net layer, assumes all input to be numpy
    inp, W = cache
    db     = np.sum(dout, axis=0)
    dW     = np.dot(inp.T, dout)
    dinp   = np.dot(dout, W.T)
    return dinp, dW, db


###################### Convolutional layer #################

def conv_layer_forward(inp, W, b):

    # This is a low key implementation, so not for performance but for readability
    # ('only' 2 x slower than optimized cython implementation though for the forward pass)
    # For faster implementation, you might want to take a look at im2col and col2im methods, and Cython implementations
    #
    # cs231n:
    # The input consists of N data points, each with C channels, height H and width
    # W. We convolve each input with F different filters, where each filter spans
    # all C channels and has height HH and width WW (HH?).
    #
    # Input:
    # - x: Input data of shape (N, C, H, W)
    # - w: Filter weights of shape (F, C, wH, wW)
    # - b: Biases, of shape (F,)
    # - conv_param: A dictionary with the following keys:
    #   - 'stride': The number of pixels between adjacent receptive fields in the
    #     horizontal and vertical directions.
    #   - 'pad': The number of pixels that will be used to zero-pad the input.
    #
    # Returns a tuple of:
    # - out: Output data, of shape (N, F, oH, oW) where H' and W' are given by
    #   oH = 1 + (H + 2 * pad - wH) / stride
    #   oW = 1 + (W + 2 * pad - wW) / stride
    # - cache: (x, w, b, conv_param)

    out = None

    # key values
    N, C, h, w = inp.shape
    F, C, Wh, Ww = W.shape

    pad = divmod(Ww, 2)[0]
    str = 1

    ph = h + 2 * pad
    pw = w + 2 * pad

    oh = int(round(1 + (ph - Wh) / str))
    ow = int(round(1 + (pw - Ww) / str))

    # padding    x_p
    x_p = np.pad(inp, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # input column    x_col (N, C, H, W)   -> (N, C*wH*wW, oH, oW)
    # retrieving correct indexes
    or2d_ind = (np.arange(Wh) * ph)[:, None] + np.arange(Ww)
    or3d_ind = (np.arange(C) * ph * pw)[:, None] + or2d_ind.ravel()
    strides = (np.arange(oh) * ph * str)[:, None] + np.arange(ow) * str
    str_ind = np.ravel(or3d_ind)[:, None] + strides.ravel()
    sam_ind = ((str_ind)[None, :] + (np.arange(N) * ph * pw * C)[:, None, None])

    x_col = np.take(x_p, sam_ind)

    # weight column    w_col (F, C, wH, wW) -> (F, C*wH*wW)
    w_col = np.reshape(W, (F, C * Wh * Ww))

    # output column    out   (N, F, oH, oW)
    out_col = np.einsum('ijk,lj->ilk', x_col, w_col) + b[None, :, None]
    out = np.reshape(out_col, (N, F, oh, ow))

    cache = (inp, x_col, W, b)
    return out, cache

def conv_layer_backward(dout, cache):
    # This is a low key implementation (using for loops), so not for performance but for readability
    # For faster implementation, you might want to take a look at im2col and col2im methods, and Cython implementations
    #
    # cs231n:
    #
    # Inputs:
    # - dout: Upstream derivatives.
    # - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    #
    # Returns a tuple of:
    # - dx: Gradient with respect to x
    # - dw: Gradient with respect to w
    # - db: Gradient with respect to b

    dx, dw, db = None, None, None

    # forward pass cache
    x, x_col, w, b = cache

    # key values
    N, C, H, W = x.shape
    F, C, wH, wW = w.shape

    pad = divmod(w.shape[-1],2)[0]
    str = 1

    pH = H + 2 * pad
    pW = W + 2 * pad

    oH = int(round(1 + (pH - wH) / str))
    oW = int(round(1 + (pW - wW) / str))

    # x gradients - for loop way
    d_col = np.reshape(dout, (N, F, oH * oW))
    w_col = np.reshape(w, (F, C * wH * wW))
    dx_col = np.sum(np.multiply(w_col[None, :, :, None], d_col[:, :, None, :]), 1)
    dx_cube = np.reshape(dx_col, (N, C, wH, wW, oH * oW))

    dx_p = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=dx_col.dtype)
    for xx in range(oW):
        for yy in range(oH):
            dx_p[:, :, yy * str:yy * str + wH, xx * str:xx * str + wW] += dx_cube[:, :, :, :, yy * oW + xx]

    dinp = dx_p[:, :, pad:-pad, pad:-pad]

    # x gradients - col2im way - about the same speed
    # ---------------------------------------------------------------------------#
    # dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)
    # dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
    # dx = col2im_indices(dx_cols, x.shape, wH, wW, pad, str)

    # x gradients - indexing way - slower
    # ---------------------------------------------------------------------------#
    # retrieving correct indexes
    # C_ind    = np.tile(np.tile(np.arange(C) [:,None], wH*wW).ravel()[:, None], oH*oW)
    #
    # H_single = np.tile(np.tile(np.arange(wH)[:,None], wW).ravel(), C)
    # H_stride = np.tile((np.arange(oH) * str)[:,None], oW ).ravel()
    # H_ind    = H_single[:, None] + H_stride
    #
    # W_single = np.tile(np.tile(np.arange(wW),wH).ravel(), C)
    # W_stride = np.tile((np.arange(oW) * str), oH ).ravel()
    # W_ind    = W_single[:, None] + W_stride
    #
    # d_col     = np.reshape(dout, (N, F, oH*oW))
    # w_col     = np.reshape(w, (F, C*wH*wW))
    # dx_col    = np.sum( np.multiply(w_col[None, :, :, None], d_col[:, :, None, :]), 1)
    #
    # dx_p      = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=dx_col.dtype)
    # np.add.at(dx_p, (slice(None), C_ind, H_ind, W_ind), dx_col)
    # dx = dx_p[:, :, pad:-pad, pad:-pad]
    # ---------------------------------------------------------------------------#

    # weight gradients
    dw_col = np.multiply(x_col[:, None, :], d_col[:, :, None, :])
    dw_sum = np.sum(dw_col, (0, 3))

    dW = np.reshape(dw_sum, (F, C, wH, wW))

    # bias gradients
    db = np.sum(dout, axis=(0, 2, 3))

    return dinp, dW, db


###################### Activation functions ################

def sigmoid_forward(un_act):
    # implements sigmoid activation function, assumes all input to be numpy
    out   = 1 / (1 + np.exp(-un_act))
    cache = un_act
    return out, cache

def sigmoid_backward(dout, cache):
    # implements derivative of sigmoid acitvation function
    un_act  = cache
    dsig, _ = sigmoid_forward(un_act) #- np.power(sigmoid_forward(un_act), 2)
    dun_act = np.multiply(dsig, dout)
    return dun_act

def softmax_forward(un_act):
    # implements softmax for prediction layer, assumes all input to be numpy
    # (no backward cause only used for last layer in combination with Cross Entropy Loss)
    exp   = np.exp(un_act)
    e_sum = np.sum(exp, axis=1, keepdims=True)
    out   = exp/e_sum
    return out


###################### 'Add-ons'  ##########################

def dropout_forward(inp, p):
    # implements dropout function, assumes all input to be numpy
    #out   = 1 / (1 + np.exp(-un_act))
    mask  = None
    mask  = np.random.rand(*inp.shape) > p
    out   = np.multiply(mask, inp) / (1 - p)
    cache = (p, mask)
    return out, cache

def dropout_backward(dout, cache):
    # implements derivative of dropout function
    p, mask = cache
    dinp = np.multiply(dout, mask) / (1 - p)
    return dinp

def batch_norm_forward(inp, gamma, beta, bn_param):
    # implements batch normalization function, assumes all input to be numpy
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = inp.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=inp.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=inp.dtype))

    out, cache = None, None
    if mode == 'train':

        inp_mean = np.sum(inp, axis=0) / N

        inp_susq = np.sum(np.power(inp, 2), axis=0)
        inp_sqsu = np.power(np.sum(inp, axis=0), 2)
        inp_var = inp_susq / N - inp_sqsu / (N * N)
        inp_std = np.sqrt(inp_var + eps)

        inp_nor = (inp - inp_mean) / inp_std

        out = gamma * inp_nor + beta

        running_mean = momentum * running_mean + (1 - momentum) * inp_mean
        running_var  = momentum * running_var + (1 - momentum) * inp_var

        cache = (inp, inp_std, inp_nor, gamma, eps)

    else:

        inp_mean = (inp - running_mean)
        inp_std = np.sqrt(running_var + eps)
        inp_nor = inp_mean / inp_std

        out = gamma * inp_nor + beta

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batch_norm_backward(dout, cache):
    # implements derivative of batch normalization function
    
    dinp, dgamma, dbeta = None, None, None
    inp, inp_std, inp_nor, gamma, eps = cache
    inp_mu = np.mean(inp, axis=0)
    N, D = inp.shape

    dinp_nor = np.multiply(dout, gamma)

    dinp = (1. / inp_std) * (
    dinp_nor - (np.sum(dinp_nor, axis=0) / N) - (inp - inp_mu) * np.sum((inp - inp_mu) * dinp_nor, axis=0) * (inp_std ** (-2.) / N))
    dgamma = np.sum(np.multiply(dout, inp_nor), axis=0)
    dbeta = np.sum(dout, axis=0)

    return dinp, dgamma, dbeta

def spatial_batchnorm_forward(inp, gamma, beta, bn_param):

    out, cache = None, None
    N, C, H, W = inp.shape
    inp_shape = np.reshape(np.transpose(inp, (0, 2, 3, 1)), (-1, C))
    out_shape, cache = batch_norm_forward(inp_shape, gamma, beta, bn_param)
    out = np.transpose(np.reshape(out_shape, (N, H, W, C)), (0, 3, 1, 2))

    return out, cache

def spatial_batchnorm_backward(dout, cache):
    # implements spatial batch normalization function for convolutional layers, assumes all input to be numpy
    dinp, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape
    dout_shape = np.reshape(np.transpose(dout, (0, 2, 3, 1)), (-1, C))
    dinp_shape, dgamma, dbeta = batch_norm_backward(dout_shape, cache)
    dinp = np.transpose(np.reshape(dinp_shape, (N, H, W, C)), (0, 3, 1, 2))

    return dinp, dgamma, dbeta


###################### Often used layer combinations  ######

def sigmoid_layer_forward(inp, W, b):
    # implements vanilla foward neural net layer, plus a sigmoid activation, assumes all input to be numpy
    un_act, fc_cache  = affine_layer_forward(inp, W, b)
    out,    sig_cache = sigmoid_forward(un_act)
    return out, (fc_cache, sig_cache)

def sigmoid_layer_backward(dout, cache):
    # implements derivatives of vanilla foward neural net layer, plus a sigmoid activation, assumes all input to be numpy
    fc_cache, sig_cache = cache
    dun_act      = sigmoid_backward(dout, sig_cache)
    dinp, dW, db = affine_layer_backward(dun_act, fc_cache)
    return dinp, dW, db

def BN_Dr_sig_layer_forward(inp, W, b, drop, gam, bet, bn_param):
    # implements vanilla foward neural net layer, then BN, then Dropout, then a sigmoid activation, assumes all input to be numpy
    un_act, fc_cache  = affine_layer_forward(inp, W, b)
    BN,     BN_cache  = batch_norm_forward(un_act, gam, bet, bn_param)
    Drop,   Dr_cache  = dropout_forward(BN, drop)
    out,    sig_cache = sigmoid_forward(Drop)
    return out, (fc_cache, BN_cache, Dr_cache, sig_cache)

def BN_Dr_sig_layer_backward(dout, cache):
    # implements derivatives of vanilla foward neural net layer, then BN, then Dropout, then a sigmoid activation, assumes all input to be numpy
    fc_cache, BN_cache, Dr_cache, sig_cache = cache
    dun_act         = sigmoid_backward(dout, sig_cache)
    dDrop           = dropout_backward(dun_act, Dr_cache)
    dBN, dgam, dbet = batch_norm_backward(dDrop, BN_cache)
    dinp, dW, db    = affine_layer_backward(dBN, fc_cache)
    return dinp, dW, db, dgam, dbet

def sig_conv_layer_forward(inp, W, b):
    # implements convolutional foward neural net layer with sigmoid activation, assumes all input to be numpy
    un_act, conv_cache  = conv_layer_forward(inp, W, b)
    out,    sig_cache   = sigmoid_forward(un_act)
    return out, (conv_cache, sig_cache)

def sig_conv_layer_backward(dout, cache):
    # implements derivatives of convolutional foward neural net layer with sigmoid activations, assumes all input to be numpy
    conv_cache, sig_cache = cache
    dun_act               = sigmoid_backward(dout, sig_cache)
    dinp, dW, db          = conv_layer_backward(dun_act, conv_cache)
    return dinp, dW, db

def softmax_layer_forward(inp, W, b):
    # implements vanilla foward neural net layer, plus a softmax, for final prediction layer, assumes all input to be numpy
    un_act, cache = affine_layer_forward(inp, W, b)
    out           = softmax_forward(un_act)
    return out, cache


###################### Loss function #######################

def cross_entropy_loss(y_pred, y_gt):
    # implements cross entropy loss and its derivative, assumes y as one hot encoded
    loss   = np.mean(np.sum((y_gt * -np.log(y_pred)), axis=1))
    d_pred = np.subtract(y_pred, y_gt)
    return loss, d_pred


###################### other ###############################

def get_random_batch(X, y, batch_size):
    # get a random batch from X and y for network training
    N      = X.shape[0]
    idx    = np.arange(N)
    np.random.shuffle(idx)
    ba_idx = idx[:batch_size]
    X_ba, y_ba = X[ba_idx], y[ba_idx]
    
    return X_ba, y_ba

def get_data_mean():
    # returns the priorly calculated means of our 150 samples data set
    return [5.84333333, 3.05733333, 3.758, 1.19933333]

def get_data_std():
    # returns the priorly calculated std's of our 150 samples data set
    return [0.82530129, 0.43441097, 1.75940407, 0.75969263]


