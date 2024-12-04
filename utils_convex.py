import os
import numpy as np
import random as rand

from ucimlrepo import fetch_ucirepo 
from sklearn.datasets import load_breast_cancer, fetch_covtype, load_diabetes

'''
General
------------------------------------------------------------------------------------------------------------------------
'''

#Helper functions for the plotting functions

def get_mean(l):
    tmp = np.array(l).T
    return [np.mean(b) for b in tmp]

def get_maxes(l):
    tmp = np.array(l).T
    return [max(b) for b in tmp]

def get_mins(l):
    tmp = np.array(l).T
    return [min(b) for b in tmp]

def get_stds(l):
    tmp = np.array(l).T
    return [np.std(b) for b in tmp]

def difference(l1,l2):
    return [a-b for a,b in zip(l1,l2)]

def addition(l1,l2):
    return [a+b for a,b in zip(l1,l2)]

def project( theta ):
    """
    This function does a projection on the ball with norm 1000 for the variable theta
    :param theta: the vector which we wish to use for projection
    :return: the projected vector
    """
    return theta 
    # Norm = theta.dot(theta)

    # if (Norm >= 1000): theta = theta / Norm

    # return theta

def Grad(X, y , theta , lambda_reg):
    """
    This function calculates the gradient
    :param X: the feature vector including at least all the features of the prefix until i
    :param y: the label vector including at least all the labels of the prefix until i
    :param theta: the current parameter vector
    :param lambda_reg: the regularization parameter
    :return: the gradient of the prefix
    """
    n = X.shape[0]
    res = np.dot(X.T, np.dot(X,theta) - y) / n
    res += lambda_reg * theta

    return res

def Loss(theta, X, y, lambda_reg):
    """
    This function calculates the value of the prefix sum of the sequence up until index k
    :param theta: the current parameter vector
    :param X: the feature vector including at least all the features of the prefix until k
    :param y: the label vector including at least all the labels of the prefix until k
    :param k: the last sample that belongs to the prefix
    :param lambda_reg: the regularization parameter
    :return: the value of the prefix sum
    """
    n = X.shape[0]
    res = np.dot((np.dot(X,theta) - y).T, np.dot(X,theta) - y) / n
    res += lambda_reg * (theta.dot(theta))
    return res

def linear_regression(X, Y, lmda):
    """
    This function is used to calculate the optimal function values for linear regression with regularization
    of the prefix sum for all prefixes of the sequence of X
    :param X: The feature vector of the entire sequence
    :param Y: The label vector of the entire sequence
    :lmda: The regularization parameter
    :return: A list with the optimal function values for all the prefixes of the sequence
    """
    x_star = None

    n,d = X.shape

    A = X.T @ X / n

    b = X.T @ Y / n

    x_star = np.linalg.solve(A + lmda*np.eye(d), b)

    f_star = Loss(x_star, X, Y, lmda)

    return x_star, f_star

def Evaluate ( Sol, Opt):
    """
    This function calculates the suboptimality gap of the function values of Sol in comparison to the optimal solutions Opt
    :param Sol: the function values of the solution vector
    :param Opt: the optimal values of the prefixes
    :return: Returns the suboptimality gap for each prefix
    """
    Res = []

    for s in Sol: Res.append( s - Opt )

    return Res

def Evaluate_runs(multiple_runs,opt):
  """
  Evaluates a list of executions of an algorithm with respect to an optimal value
  """
  multiple = [Evaluate(x,opt) for x in multiple_runs]
  return multiple

'''
Data
------------------------------------------------------------------------------------------------------------------------
'''

def data_loader(filename="breastcancer",truncate=1000):
    """
    :param filename: Select the name of the file to be loaded.(breastcancer,cod-rna,diabetes,german.numer,skin_nonskin)
    :param truncate: Select the maximum amount of data to be loaded
    :return: The feature vector and the labels for the loaded dataset
    """
    if(filename == "breastcancer"):
        # # fetch dataset 
        # breast_cancer = fetch_ucirepo(id=14) 

        # # data (as pandas dataframes) 
        # X = breast_cancer.data.features 
        # y = breast_cancer.data.targets 
        breastcancer = load_breast_cancer()
        X = breastcancer.data
        y = breastcancer.target

    else:
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target
        #X = np.array(X.todense())

    if truncate > len(X):
        X = X[:truncate]
        y = y[:truncate]


    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X = (X - mean )/ std

    # print(np.mean(X, axis=0), np.std(X, axis=0))

    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    y = (y - mean )/ std

    return X,y

'''
Algorithms 
------------------------------------------------------------------------------------------------------------------------
'''

def GD(X, y, x0, learning_rate, T, lambda_reg,X_test=None,y_test=None):
    """
    This function does gradient descent for T iterations for the first k samples
    :param X: the feature vector including at least all the features of the prefix until k
    :param y: the label vector including at least all the labels of the prefix until k
    :param x0: the intialization for the parameter vector
    :param learning_rate: the learning rate that will be used for the gradient steps
    :param T: the number of gradient steps that the function does
    :param lambda_reg: the regularization parameter
    :return: Returns an optimized parameter vector
    """
    theta = x0

    FOs=0
    FOs_list = []

    Losses = []
    Losses_test = []

    for t in range(T):
        nabla = Grad(X, y, theta, lambda_reg)
        FOs += X.shape[0]

        # print('Gradient:' , nabla)

        # print('learning_rate',learning_rate)

        theta = project(theta  - learning_rate * nabla)
        # print('diff:', np.linalg.norm(theta - theta_prev))
        loss = Loss(theta, X, y, lambda_reg)
        Losses.append(loss)
        if X_test is not None:
            loss_test = Loss(theta, X_test , y_test , 0)
            Losses_test.append(loss_test)
        FOs_list.append(FOs)
    return theta, Losses, Losses_test, FOs_list

def SVRG(X, y, x0, learning_rate, S, m, lambda_reg,X_test=None,y_test=None):
    """
    This function implements the SVRG algorithm
    :param X: the feature vector including at least all the features of the prefix until k
    :param y: the label vector including at least all the labels of the prefix until k
    :param lambda_reg: the regularization parameter
    :param S: the outer loop iterations of the SVRG algorithm that the code should execute
    :param gamma: the learning rate to be used in the step of the algorithm
    :param m: the inner loop iterations of the Svrg algorithm that the code executes
    :param k: the prefix of the samples for which we should execute SVRG
    :param x0: the initialization vector
    :return: the optimized parameter vector for the prefix k as calculated by SVRG
    """
    xs = x0
    xm = x0
    b = m

    FOs = 0

    Losses = []
    Losses_test = []
    FOs_list = []
    for s in range(S):
        nabla = Grad(X,y,xs,lambda_reg)
        FOs += X.shape[0]
        acc = 0
        for j in range(b):
            r = np.random.randint(low=0,high = X.shape[0])
            nabla_m = (np.dot(X[r],xm)-y[r]) * X[r] + lambda_reg * xm
            nabla_s = (np.dot(X[r],xs)-y[r]) * X[r] + lambda_reg * xs
            FOs += 2
            xm = project(xm - learning_rate*(nabla_m-nabla_s+nabla))
            acc= acc + xm
            Losses.append(Loss(acc/(j+1),X,y,lambda_reg))
            if X_test is not None:
                loss_test = Loss(acc/(j+1), X_test , y_test ,0)
                Losses_test.append(loss_test)
            FOs_list.append(FOs)
        xs = acc/b
        xm = xs
      #Losses.append(Loss(xs,X,y,lambda_reg))
      

    return xs,Losses, Losses_test, FOs_list

def Katyusha(X, y, x0, n, lambda_reg, S,T_in ,sigma, L, gamma):
    """
    This function implements the Katyusha algorithm
    :param X: the feature vector including at least all the features of the prefix until n
    :param y: the label vector including at least all the labels of the prefix until n
    :param x0: the initialization vector
    :param n: the prefix of the samples for which we should execute Katyusha
    :param lambda_reg: the regularization parameter
    :param S: the outer loop iterations of the katyusha algorithm that the code should execute
    :param m: the inner loop iterations of the katyusha algorithm that the code executes
    :param sigma: the strong convexity parameter for the setting
    :param L: the smoothness parameter for the setting
    :param gamma: the learning rate to be used in the step of the algorithm
    :return: the optimized parameter vector for the prefix k as calculated by katyusha
    """
    m = T_in
    t2 = 0.5
    t1 = min(np.sqrt((m*sigma)/(3*L)),0.5)
    alpha = 1/( 3* t1 *L )
    xs = x0
    ym = xs
    zm = ym
    xm = xs
    for s in range(S):
        nabla = Grad(X,y,xs,lambda_reg)
        nom = np.zeros_like(X[0])
        denom = 0
        mul = 1
        for j in range(m):
            k = s * m + j
            xm = t1 * zm + t2 * xs + (1-t1-t2) * ym
            r = rand.randint(0,n)
            nabla_m = (np.dot(X[r],xm)-y[r]) * X[r] + lambda_reg * xm
            nabla_s = (np.dot(X[r],xs)-y[r]) * X[r] + lambda_reg * xs
            grad = nabla + nabla_m - nabla_s
            zm = project(zm - alpha * grad)
            ym = project(xm - grad * gamma)
            nom = nom + mul * ym
            denom = denom + mul
            mul = mul * (1+alpha*sigma)
        xs = nom/denom
    return xs

def Produce_Katyusha(X,y,lambda_reg, S,T_in, L, gamma):
    """
    This function executes katyusha as a solver for the instance optimal problem
    :param X: the feature vector for the whole sequence
    :param y: the label vector for the whole sequence
    :param lambda_reg: the regularization parameter
    :param S: the outer loop iterations of the katyusha algorithm that the code should execute
    :param T_in: the inner loop iterations of the katyusha algorithm that the code executes
    :param L: the smoothness parameter for the setting
    :param gamma: the learning rate to be used in the step of the algorithm
    :return: Returns the function values for the point calculated by katyusha for each stage of the algorithm, as well as the FOs done
    """
    xe = np.zeros_like(X[0])
    Res = []
    FOs = []
    crFOs = 0
    for n in range(len(X)):
        xe = Katyusha(X,y,project(np.ones_like(X[0])),n ,lambda_reg, S,T_in, lambda_reg, L, gamma)
        crFOs = crFOs + S*(n+1)+T_in*S*2
        FOs.append(crFOs)
        Res.append(Loss(xe, X, y, lambda_reg))
    return Res, FOs

def SGD(X, y, theta_0, learning_rate, T, lambda_reg,X_test=None,y_test=None):
    """
    This function implements the SGD algorithm
    :param X: the feature vector including at least all the features of the prefix until k
    :param y: the label vector including at least all the labels of the prefix until k
    :param lambda_reg: the regularization parameter
    :param T: the number of iterations for which the algorithm should be executed
    :theta_0 the initialization vector
    :return: the optimized parameter vector, Losses and number of FOs
    """
    theta = theta_0

    Avg = np.zeros_like(X[0])

    FOs = 0
    FOs_list = []

    Losses = []
    Losses_test = []

    for t in range( T ):
        j = np.random.randint(low = 0,high = len(X))
        xj = X[j]
        nabla = (np.dot(xj,theta)-y[j]) * xj + lambda_reg * theta
        FOs += 1
        #gamma = 1 / (1000*(t+1) * lambda_reg )
        gamma = learning_rate * 1/np.sqrt(t+1)

        theta = project(theta  - gamma * nabla)

        #loss = Loss( Avg, X , y , lambda_reg )
        loss = Loss( theta, X , y , lambda_reg )

        Avg = Avg * (1 - 1/(t+1)) + theta / (t+1)

        Losses.append(loss)

        if X_test is not None:
            #loss_test = Loss( Avg, X_test , y_test , 0)
            loss_test = Loss( theta, X_test , y_test , 0)
            Losses_test.append(loss_test)
        FOs_list.append(FOs)
        
    return theta, Losses, Losses_test, FOs_list