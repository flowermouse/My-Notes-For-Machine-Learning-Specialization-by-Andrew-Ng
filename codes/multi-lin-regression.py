import numpy as np
import matplotlib as plt
from copy import deepcopy
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression

def predict(x, w, b):
    """
    single predict using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    return p

def compute_cost(X, y, w, b): 
    """
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape           # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    w = deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
        
    return w, b     # return final w, b

def implement(X, y):
    # initialize parameters
    initial_w = np.ones_like(X[0])
    initial_b = 1.
    # some gradient descent settings
    iterations = 10000
    alpha = 5.0e-2
    # run gradient descent 
    w_final, b_final = gradient_descent(X, y, initial_w, initial_b,
                                            compute_cost, compute_gradient, 
                                                alpha, iterations)
    print(f"b, w found by gradient descent: {b_final:0.2f},{w_final} ")
    m, _ = X.shape
    for i in range(m):
        print(f"prediction: {predict(X[i], w_final, b_final)}, target value: {y[i]}")

def normalize(X):
    """
    z-score normalization
    """
    # avoid division by zero
    eps = 1e-8
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - np.mean(X[:, i])) / (np.std(X[:, i]) + eps) 

def main():
    boston = fetch_openml(name='boston', version=1)
    X = boston.data.to_numpy(dtype=np.float32)[:10, :]
    normalize(X)
    print(X)
    y = boston.target.to_numpy(dtype=np.float32)[:10]
    implement(X, y)

    # compare with sklearn
    model = LinearRegression()
    model.fit(X, y)
    print(f"b, w found by sklearn: {model.intercept_},{model.coef_} ")
    m, _ = X.shape
    for i in range(m):
        print(f"prediction: {model.predict([X[i]])}, target value: {y[i]}")
    
main()