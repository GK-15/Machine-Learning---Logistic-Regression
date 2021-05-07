import numpy as np
import matplotlib.pyplot as plt


def plot_boundary(X, theta, ax1):
    
    #min_x1 = 0.0
    #max_x1 = 0.0
    #x2_on_min_x1 = 0.0
    #x2_on_max_x1 = 0.0
   
    #########################################
    # Write your code here
    # Re-arrange the terms in the equation of the hypothesis function, and solve with respect to x2, to find its values on given values of x1
    min_x1=min(X[:,1])
    max_x1=max(X[:,1])
    a=X[:,1]
    x2_on_min_x1= X[np.argmin(a),2]
    x2_on_max_x1= X[np.argmax(a),2]
    
    ########################################/
    
    x_array = np.array([min_x1, max_x1])
    y_array = np.array([x2_on_min_x1, x2_on_max_x1])
    ax1.plot(x_array, y_array, c='black', label='decision boundary')
    
    # add legend to the subplot
    ax1.legend()
    
