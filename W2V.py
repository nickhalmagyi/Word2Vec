
# coding: utf-8

# In[1]:

## Runs my implementation of Word2Vec
## The input is a text file plus some parameters 
## The pre-processing is performed by functions in TexttoDict

import numpy as np
import theano
import theano.tensor as T
import nltk
import string
from collections import defaultdict
from TexttoDict import *

rng = np.random


# N-- size of hidden layer
# inarr, out arr are the output of run_container

def W2Vtrain(inarr, outarr, N):
# length of cutoff dictionary
    V=len(inarr[0])
    
    
# (x,y) are shared variables for (input, output)
    x = T.dvector("x") # a vector
    y = T.dvector("y") # a vector


# (W1,W2) are the weight matrices for input-hidden and hidden-output
    W1 = theano.shared(rng.randn(V,N), name="W1")   # a V x N matrix
    W2 = theano.shared(rng.randn(N,V), name="W2")   # a N x V matrix

# This evaluates the hidden and output layers
# This is a linear operation, no logistic function unlike neural nets
    vhidden = T.dot(x, W1)   
    vout = T.dot(vhidden,W2)   

# The prediction thresholded
   # prediction = p_2 > 0.5                    

# Entropy loss function (this is a scalar)
# y is the target word. x in the average of the bag
    Entropy = T.dot(y,vout) - T.log((T.exp(vout)).sum())

# Compute the gradient of the cost 
# w.r.t weight vector w and bias term b
# These will be updated on each training run
    gradW1, gradW2 = T.grad(Entropy, [W1, W2])

    train = theano.function(
          inputs=[x,y],
          outputs=[Entropy],
          updates=((W1, W1 - 0.1 * gradW1),(W2, W2 - 0.1 * gradW2)))

# Training Phase
# During the training phase w and b are being updated which are then used in "prediction"
# So D[0] and D[1] are not updated but the parameters in the function "train" 
# are updated at each step

    for i in range(len(inarr)):
        train(inarr[i], outarr[i])
    
    
    return [W1.get_value(),Entropy]




