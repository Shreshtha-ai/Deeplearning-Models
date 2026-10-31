import numpy as np
import pandas as pd
df = pd.DataFrame([[8,8,4],[7,9,5],[6,10,6],[5,12,7]], columns=['cgpa', 'profile_score', 'lpa'])

def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):

        parameters['W' + str(l)] = np.ones((layer_dims[l-1], layer_dims[l]))*0.1        
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)  )
        

    return parameters




initialize_parameters([2,2,1])


def linear_forward(A_prev, W, b):
  
  Z = np.dot(W.T, A_prev) + b
  
  return Z



