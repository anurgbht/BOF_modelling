from __future__ import print_function
import numpy as np
import pandas as pd
import random
import pylab as plt
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn import preprocessing
import h5py
import time

################################################################################
################################################################################
################################################################################

write_path = "D:\\OneDrive - Tata Insights and Quants, A division of Tata Industries\\Confidential\\Projects\\Steel\\LD2 BOF\\data\\jan17-oct17\\results\\"

def my_plot(pred_test,pred_train,y_test,y_train):

    y_test = list(y_test)
    y_train = list(y_train)
    err_test = [x-y for x,y in zip(pred_test,y_test)]
    err_train = [x-y for x,y in zip(pred_train,y_train)]

    plt.subplot(2,2,1)
    plt.plot(pred_test,y_test,'.')
    plt.ylabel('Actual')
    plt.minorticks_on()
    plt.grid(1,'both')
    plt.title('Test RMS : '+str(np.sqrt(np.mean(np.square(err_test)))))

    plt.subplot(2,2,2)
    plt.plot(pred_train,y_train,'.')
    plt.minorticks_on()
    plt.grid(1,'both')
    plt.ylabel('Actual')
    plt.title('Train RMS : '+str(np.sqrt(np.mean(np.square(err_train)))))

    plt.subplot(2,2,3)
    plt.hist(err_test,bins = 150)
    plt.minorticks_on()
    plt.grid(1,'both')
    plt.title('Test : $\pm$5 : {0:.3f}% $\pm$15 : {1:.3f}% $\pm$25 : {2:.3f}%'.format(100*sum([np.abs(x)<5 for x in err_test])/len(err_test),100*sum([np.abs(x)<15 for x in err_test])/len(err_test),100*sum([np.abs(x)<25 for x in err_test])/len(err_test)))
    
    plt.subplot(2,2,4)
    plt.hist(err_train, bins = 150)
    plt.minorticks_on()
    plt.grid(1,'both')
    plt.title('Train : $\pm$5 : {0:.3f}% $\pm$15 : {1:.3f}% $\pm$25 : {2:.3f}%'.format(100*sum([np.abs(x)<5 for x in err_train])/len(err_train),100*sum([np.abs(x)<15 for x in err_train])/len(err_train),100*sum([np.abs(x)<25 for x in err_train])/len(err_train)))
    plt.show()

def result_plot(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.minorticks_on()
    plt.grid(1,'both')
    plt.show()

# define base model
# for practise purposes only
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def save_model(model):
    model_json = model.to_json()
    with open(write_path + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(write_path + "model.h5")
    print("Saved model to disk")

def print_structure(weight_file_path):
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()

def weight_writer(model):
    print("Writing the weights to a file")
    writer = pd.ExcelWriter(write_path + 'ANN_weights.xlsx')
    for i in range(len(model.get_config())):
        wts = list(model.layers[i].get_weights()[0])
        bs = list(model.layers[i].get_weights()[1])
        wts.append(bs)
        wts = pd.DataFrame(wts)
        wts.to_excel(writer,sheet_name = 'layer_'+str(i+1))
    writer.save()
    print("Done writing the weights !")

# try generalizing the function definition for the activation function
# right now we have relu
# more needs to be included and the code needs to be modular
def relu(x):
    if x>0:
        tt = x
    else:
        tt = 0
    return tt

def apply_activation(mat_dat,bs):
    mat_dat = add_bias(mat_dat,bs)
    original_shape = mat_dat.shape
    temp = mat_dat.flatten().tolist()[0]
    temp = [relu(x) for x in temp]
    temp = np.matrix(temp).reshape(original_shape)
    return temp

def add_bias(mat_dat,bs):
    for i in range(mat_dat.shape[0]):
        mat_dat[i] = mat_dat[i] + bs
    return mat_dat

def my_predict(X,model):
    X  = np.matrix(X)
    for i in range(len(model.get_config())):
        # let us first ignore the bias and do matrix multiplication straight away
        wts = np.matrix(model.layers[i].get_weights()[0])
        bs = model.layers[i].get_weights()[1]
        X = apply_activation(X*wts,bs)
    return X

################################################################################
#######################MODEL BUILDING FUNCTIONS HERE###########################
################################################################################

def prelim_keras(*args):
    print('Preliminary Keras-ANN Analysis')
    proceed_flag = 0
    if len(args) == 3:
        X = args[0]
        y = args[1]
        nb_epoch,batch_size,verbose,plot_flag = args[2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        proceed_flag = 1
    if len(args) == 5:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        nb_epoch,batch_size,verbose,plot_flag = args[4]
        proceed_flag = 1

    if proceed_flag == 1:
        input_dim  = X_train.shape[1]
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model = Sequential()
        model.add(Dense(58, input_dim=input_dim, init='normal', activation='relu'))
        model.add(Dense(23, init='normal', activation='relu'))
        model.add(Dense(1,init='normal'))
        # Compile clf
        model.compile(loss='mse', optimizer='adam')

        # Fit the clf
        history = model.fit(X_train, y_train,nb_epoch=nb_epoch, batch_size=batch_size,  verbose=verbose)

        # calculate predictions
        X_test = np.array(X_test)
        pred_test = model.predict(X_test)
        pred_test = [x[0] for x in pred_test]
        
        pred_train = model.predict(X_train)
        pred_train = [x[0] for x in pred_train]
        
        if plot_flag == 1:
            my_plot(pred_test,pred_train,y_test,y_train)
            result_plot(history)

        return model

    else:
        print('Something went wrong')

def prelim_linear(*args):
    print('Preliminary Keras Analysis')
    proceed_flag = 0
    if len(args) == 2:
        X = args[0]
        y = args[1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        proceed_flag = 1
    if len(args) == 4:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        proceed_flag = 1

    if proceed_flag == 1:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        clf = linear_model.LinearRegression()
        # Fit the clf
        clf.fit(X_train, y_train)

        # calculate predictions
        X_test = np.array(X_test)
        pred_test = clf.predict(X_test)
        pred_train = clf.predict(X_train)

        my_plot(pred_test,pred_train,y_test,y_train)

    else:
        print('Something went wrong')
