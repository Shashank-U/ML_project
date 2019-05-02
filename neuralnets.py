# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:19:53 2019

@author: USER
"""

import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def derivative_of_sigmoid(x):
    return x * (1.0 - x)

class NeuralNet:
    def __init__(self, x, y):
        self.input      = x
        self.wts1   = np.random.rand(self.input.shape[1],4) 
        self.wts2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def forwardprop(self):
        self.L1 = sigmoid(np.dot(self.input, self.wts1))
        self.output = sigmoid(np.dot(self.L1, self.wts2))

    def backwardprop(self):
        #  chain rule to find derivatives wrt wts1 and wts2
        derived_wts2 = np.dot(self.L1.T, (2*(self.y - self.output) * derivative_of_sigmoid(self.output)))
        derived_wts1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * derivative_of_sigmoid(self.output), self.wts2.T) * derivative_of_sigmoid(self.L1)))

        # updating the weights
        self.wts1 = self.wts1 + derived_wts1
        self.wts2 = self.wts2 + derived_wts2


if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    NN = NeuralNet(X,y)

    for i in range(1500):
        NN.forwardprop()
        NN.backwardprop()

    print(NN.output)