# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 15:10:44 2018
@author: M@nu
"""
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras import backend as K
from keras.models import Model
import numpy as np
import random

epoch = 6

'''Compute the euclidean distance bwtween vector x and y.'''
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


'''Apply a contrastive loss function.'''
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


'''Create positive and negative pairs'''
def create_positive_negatives(x_train,x_test,y_train,y_test,x_val,y_val):
    # create training+test positive and negative pairs
    digit_indices_train = [np.where(y_train == i)[0] for i in [2,3,4,5,6,7]]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices_train)
    
    digit_indices_test = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(x_test, digit_indices_test)
    
    digit_indices_val = [np.where(y_val == i)[0] for i in [2,3,4,5,6,7]]
    val_pairs, val_y = create_pairs(x_val, digit_indices_val)
    
    return tr_pairs,tr_y,te_pairs,te_y,val_pairs,val_y


'''Distribute balanced pairs'''
def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    for d in range(len(digit_indices)):       
        for i in range(len(digit_indices[d])-1):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            dn = random.choice([idx for idx in range(len(digit_indices)) if idx != d])
            j = random.randint(0,len(digit_indices[dn])-1)
            z1, z2 = digit_indices[d][i], digit_indices[dn][j]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


'''Create base network'''
def create_base_network_cnn(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(2048, activation='softmax')(x)
    return Model(input,x)

'''Compute accuracy'''
def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)
def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

''' Import images digits dataset from MNIST '''
def import_mnist_split():
    #Import Mnist dataset
    (x_train_in, y_train_in), (x_test_in, y_test_in) = mnist.load_data()
    
    #Concatenate train and test from Mnist
    x = np.concatenate((x_train_in,x_test_in),axis=0)
    y = np.concatenate((y_train_in,y_test_in),axis=0)
    
    #Extract Indexes 2 to 7
    digits_2_to_7 = [2,3,4,5,6,7]
    ix_digits_2_to_7 = np.isin(y,digits_2_to_7)
    
    #Extract Indexes 0,1,8,9
    digits_0_1_8_9 = [0,1,8,9]
    ix_digits_0_1_8_9 = np.isin(y,digits_0_1_8_9)
    
    #Extract digits 2 to 7
    x_digits_2_to_7 = x[ix_digits_2_to_7]
    y_digits_2_to_7 = y[ix_digits_2_to_7]
    
    #Extract values 0,1,8,9
    x_digits_0_1_8_9 = x[ix_digits_0_1_8_9]
    y_digits_0_1_8_9 = y[ix_digits_0_1_8_9]
    
    x_digits_2_to_7_train, x_digits_2_to_7_test, y_digits_2_to_7_train, y_digits_2_to_7_test = train_test_split(x_digits_2_to_7, y_digits_2_to_7, test_size=0.2)
    
    #Assign values train and validation
    x_train = x_digits_2_to_7_train
    y_train = y_digits_2_to_7_train
    x_train,x_val,y_train,y_val = train_test_split(x_train, y_train, test_size=0.2)
    
    #Assign values test
    x_test = np.concatenate((x_digits_0_1_8_9,x_digits_2_to_7_test),axis=0)
    y_test = np.concatenate((y_digits_0_1_8_9,y_digits_2_to_7_test),axis=0)
    
    return x_train, y_train, x_test, y_test,x_val,y_val

''' Preprocessing of images'''
def pre_process_data_conv():
    # the data, split between train and test sets
    x_train, y_train, x_test, y_test, x_val, y_val = import_mnist_split()

    # Normalize pixels between 0 - 1
    x_train = x_train/ 255
    x_test = x_test / 255
    x_val = x_val / 255
    
    #Adjust Dimensions
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_val = np.expand_dims(x_val, -1)
        
    # Explicity state input shape
    length = x_train.shape[1]
    width = x_train.shape[2]
    channels = x_train.shape[3]
    
    # Create and compile model
    input_shape = (length, width, channels)
    
    return x_train,x_test,y_train,y_test,x_val,y_val,input_shape


''' Creating general structure of the model '''
def create_model_cnn(input_shape):
    base_network = create_base_network_cnn(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    siames_a = base_network(input_a)
    siames_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([siames_a, siames_b])
    
    model = Model([input_a, input_b], distance)
    
    return model

''' Fitting the model'''
def fit_model(model,tr_pairs,tr_y,te_pairs,te_y,val_pairs,val_y):
    # train
    optimizer = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy])
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              epochs=epoch,
              validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_y),
              verbose = True)
    
    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)
    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    return history

''' Calculating the Error trends'''
def Error_trends(history,metric_name):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    plt.plot(range(1,epoch+1),metric, label = 'Train ' + metric_name)
    plt.plot(range(1,epoch+1),val_metric, label = 'Validation ' + metric_name)
    plt.title("Comparing training and validation")
    plt.xlabel("Epchos")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()    


#CONVOLUTIONAL
# the data, split between train and test sets
x_train,x_test,y_train,y_test,x_val,y_val,input_shape = pre_process_data_conv()
# create training+test positive and negative pairs
tr_pairs,tr_y,te_pairs,te_y,val_pairs,val_y = create_positive_negatives(x_train,x_test,y_train,y_test,x_val,y_val)
# network definition
model = create_model_cnn(input_shape)
#Fit Model
history = fit_model(model,tr_pairs,tr_y,te_pairs,te_y,val_pairs,val_y)
#Plot trends
Error_trends(history,'loss')
