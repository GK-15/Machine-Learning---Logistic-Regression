import numpy as np
import math


def sigmoid(z):
    
    #output = 0.0
    #########################################
    # Write your code here

    output= 1 /(1+(math.e**-z))
    # modify this to return z passed through the sigmoid function

    ########################################/
    
    return output