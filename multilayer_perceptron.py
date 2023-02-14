#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 20:10:11 2022

@author: raksha
"""


#import library
import numpy as np

class MultilayerPerceptron:
    
    #define the weight and bias variables
    def __init__(self, w1, b1, w2, b2, w3, b3):
        self.net = {}
        self.net['w1'] = w1
        self.net['b1'] = b1

        self.net['w2'] = w2
        self.net['b2'] = b2  
        
        self.net['w3'] = w3
        self.net['b3'] = b3
        
     #define sigmoid activation function   
    def sigmoid(self, a):
        return 1/(1 + np.exp(-a))
    
    #define forward function for dot and sum operation for each node
    def forward(self, x):
        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        b1, b2, b3 = self.net['b1'], self.net['b2'], self.net['b3']
        
        a1 = np.dot(x, w1) + b1     
        z1 = self.sigmoid(a1)
        
                
        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)
        
                
        a3 = np.dot(z2, w3) + b3
        z3 = self.sigmoid(a3)
        
        return z3
    
if __name__ == '__main__':
    print("Implement Multilayer Perceptron by invoking class MultilayerPerceptron")
    
        