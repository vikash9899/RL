# import the libray's
from __future__ import division, print_function
import numpy as np
import pylab as pl

# markov chains 
def markov_chains(initial_prob, after_steps):
    result = np.identity(3)
    for i in range(after_steps):
        result = np.dot(result, initial_prob)
    
    return result

# main funtion where program starts 
def main():
    P = np.array([[0.90, 0.05, 0.05],
                [0.10, 0.70, 0.20],
                [0, 0, 1]])
    
    result = markov_chains(P, 2)
    print(result)
    

if __name__ == '__main__':
    main()



