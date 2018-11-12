# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:33:53 2018

@author: corve
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn import model_selection
import random
# ----------------------------------------------------------------------------

def differential_evolution(fobj, 
                           bounds, 
                           mut=2, 
                           crossp=0.7, 
                           popsize=20, 
                           maxiter=100,
                           verbose = False):
    '''
    This generator function yields the best solution x found so far and 
    its corresponding value of fobj(x) at each iteration. In order to obtain 
    the last solution,  we only need to consume the iterator, or convert it 
    to a list and obtain the last value with list(differential_evolution(...))[-1]    
    
    
    @params
        fobj: function to minimize. Can be a function defined with a def 
            or a lambda expression.
        bounds: a list of pairs (lower_bound, upper_bound) for each 
                dimension of the input space of fobj.
        mut: mutation factor
        crossp: crossover probability
        popsize: population size
        maxiter: maximum number of iterations
        verbose: display information if True    
    '''
    def denormalized (vec,bounds):
        bound_mins = np.asarray(bounds).T[0]
        bound_maxs = np.asarray(bounds).T[1]
        diff = np.fabs(bound_mins- bound_maxs)
        den_vec = bound_mins + vec * diff
        return den_vec
        
        
    ##INITIALIZING
    #Create a randomic initial population with numbers from 0 to 1
    n_dimensions = len(bounds)
    init_popul_norm = np.random.rand(popsize, n_dimensions)      
    init_popul_denorm = denormalized(init_popul_norm,bounds)
    
    
    #Evaluate the function fobj with each individual of the population to get the best of the population.
    cost = np.asarray([fobj(individual) for individual in init_popul_denorm])
    best_idx = np.argmin(cost)
    best = init_popul_denorm[best_idx]
    
    ##MUTATION AND RECOMBINATION
    if verbose:
        print('** Lowest cost in initial population = {} '.format(cost[best_idx]))        
    for i in range(maxiter):
        if verbose:
            print('** Starting generation {}, '.format(i))        

            
        for j in range(popsize):
            #MUTATION
            #Defining the individuals a,b,c
            indexes = [idx for idx in range(popsize) if idx != j]
            points = random.sample(indexes, 3)
            a= init_popul_norm[points[0]]
            b= init_popul_norm[points[1]]
            c= init_popul_norm[points[2]]
            w= init_popul_norm[j]
            
            #Creating the mutant vector
            mutant_vector = a + mut * (b - c)
            
            #In case the new vector don't be normalized, we adjust the values.
            mutant_vector = np.clip(mutant_vector, 0, 1)          
                       
            #RECOMBINATION
            #Merge the mutant vector normalized and the init popul normalized             
            recomb_points = np.random.rand(n_dimensions) < crossp
            if not np.any(recomb_points):
                recomb_points[np.random.randint(0, n_dimensions)] = True 
            trial = np.where(recomb_points, mutant_vector, w)
        
            #REPLACEMENT
            #Apply the denormalization process 
            cost_trial = fobj(denormalized(trial,bounds))
            cost_w = fobj(denormalized(w,bounds))            
           
            if cost_trial < cost_w:
                cost[j] = cost_trial
                init_popul_norm[j] = trial  
            
            if cost_trial < cost[best_idx]:
                best_idx = j
                best = denormalized(trial,bounds)
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print('<<<<<<<<< New best {} found at generation {} >>>>>>>>>'.format(abs(cost[best_idx]),i)) 
        
        yield best, cost[best_idx]


# ----------------------------------------------------------------------------

def task_1():
    '''
    Our goal is to fit a curve (defined by a polynomial) to the set of points 
    that we generate. 
    '''

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    def fmodel(x, w):
        '''
        Compute and return the value y of the polynomial with coefficient 
        vector w at x.  
        For example, if w is of length 5, this function should return
        w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 
        The argument x can be a scalar or a numpy array.
        The shape and type of x and y are the same (scalar or ndarray).
        '''
        if isinstance(x, float) or isinstance(x, int):
            y = 0
        else:
            assert type(x) is np.ndarray
            y = np.zeros_like(x)
        for i in reversed(range(0,len(w))):
            y = w[i] + y*x        
       
        return y

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
    def rmse(w):
        '''
        Compute and return the root mean squared error (RMSE) of the 
        polynomial defined by the weight vector w. 
        The RMSE is is evaluated on the training set (X,Y) where X and Y
        are the numpy arrays defined in the context of function 'task_1'.        
        '''
        Y_pred = fmodel(X, w)
        return np.sqrt(sum((Y - Y_pred)**2)/len(Y))


    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    
    # Create the training set
    X = np.linspace(-5, 5, 500)
    Y = np.cos(X) + np.random.normal(0, 0.2, len(X))
    
    # Create the DE generator
    de_gen = differential_evolution(rmse, [(-5, 5)] * 6,popsize= 20, mut=1, maxiter=2000)
    
    #Defining the cost target and finishing until getting it.
    cost_target = 0.3   
    for i , p in enumerate(de_gen):
        w, c_w = p            
        if c_w< cost_target:
            break
        
    # Print the search result
    print('Stopped search after {} generation. Best cost found is {}'.format(i,c_w))
        
    # Plot the approximating polynomial
    plt.scatter(X, Y, s=2)
    plt.plot(X, np.cos(X), 'r-',label='cos(x)')
    plt.plot(X, fmodel(X, w), 'g-',label='model')
    plt.legend()
    plt.title('Polynomial fit using DE')
    plt.show()    

# ----------------------------------------------------------------------------

def task_2():
    '''
    Goal : find hyperparameters for a MLP
    
       w = [nh1, nh2, alpha, learning_rate_init]
    '''
    
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP with trained 
        with the hyperparameter vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        '''
        
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1
                int(1+w[1]), # nh2
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )


        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=False, tol=1e-4, random_state=1
                            )
        
        clf.fit(X_train_transformed, y_train)
        
        # compute the accurary on the test set
        mean_accuracy = clf.score(X_test_transformed, y_test)
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset_targets.txt',dtype=np.uint8)[:1000]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)


    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
    
    de_gen = differential_evolution(
            eval_hyper, 
            bounds, 
            mut = 1,
            popsize=20, 
            maxiter=20,
            verbose=True)
    
    for i, p in enumerate(de_gen):
        w, c_w = p
        print('Generation {},  best cost {}'.format(i,abs(c_w)))
        # Stop if the accuracy is above 90%
        if abs(c_w)>0.90:
            break
 
    # Print the search result
    print('Stopped search after {} generation. Best accuracy reached is {}'.format(i,abs(c_w)))   
    print('Hyperparameters found:')
    print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
    print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
    
# ----------------------------------------------------------------------------

def task_3():
    pass
    
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP with trained 
        with the hyperparameter vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        '''
        
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1
                int(1+w[1]), # nh2
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )


        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=False, tol=1e-4, random_state=1
                            )
        
        clf.fit(X_train_transformed, y_train)
        # compute the accurary on the test set
        mean_accuracy = clf.score(X_test_transformed, y_test)
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset_targets.txt',dtype=np.uint8)[:1000]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters

    dicts = {}
    accuracies = np.array([])
    generations = np.array([])
    #mutants = np.arange(0.1,2.5,0.1)
    #individuals = np.arange(4,100,1)
    experiment =np.array([[5,40],[10,20],[20,10],[40,5]])

    for po,ma in experiment:
        de_gen = differential_evolution(eval_hyper, bounds, mut = 1, popsize=po, maxiter=ma, verbose=True)
        for i, p in enumerate(de_gen):
            w, c_w, gen = p
            print('Generation {},  best cost {}'.format(i,abs(c_w)))
            # Stop if the accuracy is above 90%
            if abs(c_w)>0.90:
                break
        dicts[(po,ma)] = abs(c_w)
        accuracies = np.append(accuracies,c_w)
        generations = np.append(generations,gen)
    print(dicts)
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    pass
    task_1()    
#   task_2()    
#   task_3()    

