#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import *
from keras.callbacks import *
import os
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from commons import mean_absolute_percentage_error
from keras.layers import *
from sklearn.pipeline import Pipeline
from keras.utils import *
from tensorflow.keras.models import load_model
from keras.optimizers import *
from keras.wrappers.scikit_learn import KerasClassifier


#%%

np.random.seed(7)

#%%

# load dataset
dataframe = pd.read_csv(path_PCA_output + "pca_75_clas.csv", sep=',')

#%%

dataframe.head(3)

#%%

dataframe.shape

#%%

length=dataframe.shape[1]-1

#%%

length

#%%

# split into input (X) and output (Y) variables
X = dataframe.iloc[:,0:length]
y = dataframe['priceUSD']

#%%

X.head(3)

#%%

y=np.ravel(y)

#%%

y

#%%

shape=X.shape[1]

#%%



#%%

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, train_size=0.8, shuffle=False, random_state=7)

#%%

estimators=[]

#%%

estimators.append(['robust',RobustScaler()])

#%%

estimators.append(['mixmax',MinMaxScaler()])

#%%

scale=Pipeline(estimators,verbose=True)

#%%

scale.fit(X_train)

#%%

X_train=scale.transform(X_train)

#%%



#%%

X_test=scale.transform(X_test)

#%%

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-2
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

#%%

# define neural network model
def sequential_model(initializer='normal', activation='relu', neurons=300, NUM_FEATURES=shape):
    # create model
    model = Sequential()
    model.add(Dense(400, input_shape=(NUM_FEATURES,), kernel_initializer=initializer, activation=activation))
    model.add(Dense(500, activation=activation))
    model.add(Dense(100, activation=activation))
    model.add(Dense(2, activation='softmax', kernel_initializer=initializer))
    # Compile model
    adam=Adam(lr=lr_schedule(0),amsgrad=True)
    #sgd=keras.optimizers.SGD(learning_rate=0.08, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

#%%



#%%

mcp_save = ModelCheckpoint('trained_models/ANN_cls_interval3_pca.hdf5', save_best_only=True, monitor='val_loss', mode='max')
earlyStopping = EarlyStopping(monitor='val_loss', patience=100,verbose=1, mode='max')

#%%

classifier=KerasClassifier(build_fn=sequential_model,batch_size=32, epochs=1000,validation_split=0.1,validation_freq=1, shuffle=True,use_multiprocessing=True, callbacks=[mcp_save,earlyStopping])

#%%



#%%

classifier.fit(X_train,y_train)

#%%

prediction_model = load_model('trained_models/ANN_cls_interval3_pca.hdf5',compile=False)

#%%

y_pred = prediction_model.predict_classes(X_test)

#%%

acc=accuracy_score(y_test,y_pred)
acc

#%%

f1=f1_score(y_test,y_pred,average='weighted')
f1

#%%

auc=roc_auc_score(y_test,y_pred)
auc

#%%

y_prob=[prediction_model.predict(X_test).max() for i in range(len(y_test))]

#%%

print(classification_report(y_test,y_pred,labels=[0,1], target_names=['decrease','increase']))

#%%

predictions=pd.DataFrame(zip(np.ravel(y_test),np.ravel(y_pred)),columns=['y_test','y_pred'])

#%%

predictions

#%%



#%%


