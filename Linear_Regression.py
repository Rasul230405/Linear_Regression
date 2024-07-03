import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.__W = []
        self.__cost_history = []

    def __cost_function(self, X, Y, W):
        h = X.dot(W)
        J = np.sum(((h - Y)**2) / (2 * len(Y)))
        return J

    def __format_input(self, X):    
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
       
        new_col = np.ones((X.shape[0], 1))
        new_X = np.hstack((new_col, X))

        return new_X
        
    def __gradient_descent(self, X, Y, alpha=0.01, iteration=1000):
        
        X = self.__format_input(X)
        Y = np.array(Y)
        W = np.zeros(X.shape[1])  
        
        for i in range(iteration):
            h = X.dot(W)
            diff = h - Y
            gradient = X.T.dot(diff) / len(Y)
            W = W - alpha * gradient  
            
            cost = self.__cost_function(X, Y, W)
            self.__cost_history.append(cost)
        return W
        
    def fit(self, X, Y):
          
        X = np.array(X)
        Y = np.array(Y)  
        result = self.__gradient_descent(X, Y, 0.01)
        self.__W = result
        
    def predict(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
    
        new_X = self.__format_input(X)
        h = new_X.dot(self.__W)
        return h
        
    def plot_costFunction(self):
        plt.plot(self.__cost_history)
