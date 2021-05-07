import numpy as np
from calculate_hypothesis import *




def compute_cost(X, y, theta):
    # initialize cost
    J = 0.0
    # get number of training examples
    m = y.shape[0]
    
    # Compute cost for logistic regression.
    for i in range(m):
        hypothesis = calculate_hypothesis(X, theta, i)
        output = y[i]
        #cost = 0.0
        #########################################
        # Write your code here
        cost = -(1 / m) * np.sum(y * np.log(probability(theta, X)) + (1 - y) * np.log(1 - probability(theta, X)))
        # You must calculate the cost
        
        ########################################/
        J += cost
    J = J/m
    
    return J

