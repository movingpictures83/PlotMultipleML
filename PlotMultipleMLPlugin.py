#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from pandas import concat
from pandas import read_csv
from helper import series_to_supervised, stage_series_to_supervised


# In[3]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,7'

import PyPluMA
import PyIO

class PlotMultipleMLPlugin:
 def input(self, inputfile):
     self.parameters = PyIO.readParameters(inputfile)

 def run(self):
     pass

 def output(self, outputfile):
  #revised = pd.read_csv('../data/response.csv', index_col=0)
  revised = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["data"])
  revised.fillna(0, inplace=True)

  # specify the number of lag hours
  n_hours = 24*7
  K = 24
  stages = revised[['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
  stages_supervised = series_to_supervised(stages, n_hours, K)
  non_stages = revised[['WS_S4', 'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26', 'PUMP_S25B', 'PUMP_S26', 'MEAN_RAIN']]
  non_stages_supervised = series_to_supervised(non_stages, n_hours-1, 1)
  non_stages_supervised_cut = non_stages_supervised.iloc[24:, :]
  n_features = stages.shape[1] + non_stages.shape[1]
  non_stages_supervised_cut.reset_index(drop=True, inplace=True)
  stages_supervised.reset_index(drop=True, inplace=True)

  all_data = concat([
                   non_stages_supervised_cut.iloc[:, :],
                   stages_supervised.iloc[:, :]],
                   axis=1)
  print("all_data.shape:", all_data.shape)

  all_data = all_data.values
  n_train_hours = int(len(all_data)*0.8)
  test = all_data[76968:, :]

  n_obs = n_hours * n_features
  test_X, test_y = test[:, :n_obs], test[:, -stages.shape[1]*K:]

  scaler = MinMaxScaler(feature_range=(0, 1))
  test_X = scaler.fit_transform(test_X)
  test_y = scaler.fit_transform(test_y)

  test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

  model_mlp = keras.models.load_model(PyPluMA.prefix()+'/saved_model/mlp.h5')
  model_rnn = keras.models.load_model(PyPluMA.prefix()+'/saved_model/rnn.h5')
  model_lstm = keras.models.load_model(PyPluMA.prefix()+'/saved_model/lstm.h5')
  model_cmlp = keras.models.load_model(PyPluMA.prefix()+'/saved_model/cnn.h5')
  model_crnn = keras.models.load_model(PyPluMA.prefix()+'/saved_model/rcnn.h5')

  yhat_mlp = model_mlp.predict(test_X)
  yhat_rnn = model_rnn.predict(test_X)
  yhat_lstm = model_lstm.predict(test_X)
  yhat_cmlp = model_cmlp.predict(test_X)
  yhat_crnn = model_crnn.predict(test_X)
  inv_yhat_mlp = scaler.inverse_transform(yhat_mlp)
  inv_yhat_rnn = scaler.inverse_transform(yhat_rnn)
  inv_yhat_lstm = scaler.inverse_transform(yhat_lstm)
  inv_yhat_cmlp = scaler.inverse_transform(yhat_cmlp)
  inv_yhat_crnn = scaler.inverse_transform(yhat_crnn)
  inv_yhat_mlp = inv_yhat_mlp[-1:, :]
  inv_yhat_rnn = inv_yhat_rnn[-1:, :]
  inv_yhat_lstm = inv_yhat_lstm[-1:, :]
  inv_yhat_cmlp = inv_yhat_cmlp[-1:, :]
  inv_yhat_crnn = inv_yhat_crnn[-1:, :]

  inv_yhat_mlp_reshape = np.reshape(inv_yhat_mlp, (-1,4))
  inv_yhat_rnn_reshape = np.reshape(inv_yhat_rnn, (-1,4))
  inv_yhat_lstm_reshape = np.reshape(inv_yhat_lstm, (-1,4))
  inv_yhat_cmlp_reshape = np.reshape(inv_yhat_cmlp, (-1,4))
  inv_yhat_crnn_reshape = np.reshape(inv_yhat_crnn, (-1,4))

  #refer = pd.read_csv('../data/ras response.csv', index_col=0)
  refer = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["reference"], index_col=0)
  import seaborn as sns

  df = pd.DataFrame(columns = ['OBS', 'RAS', 'MLP', 'RNN', 'LSTM', 'CMLP', 'CRNN'])
  iloc1 = int(self.parameters["iloc1"])
  iloc2 = int(self.parameters["iloc2"])
  inv_yhat = int(self.parameters["inv_yhat"])
  df.iloc[:, 0] = refer.iloc[:, iloc1]
  df.iloc[:, 1] = refer.iloc[:, iloc2]
  df.iloc[:, 2] = inv_yhat_mlp_reshape[:, inv_yhat]
  df.iloc[:, 3] = inv_yhat_rnn_reshape[:, inv_yhat]
  df.iloc[:, 4] = inv_yhat_lstm_reshape[:, inv_yhat]
  df.iloc[:, 5] = inv_yhat_cmlp_reshape[:, inv_yhat]
  df.iloc[:, 6] = inv_yhat_crnn_reshape[:, inv_yhat]

  plt.rcParams["figure.figsize"] = (16, 5)

  LINEWIDTH = 2.5
  plt.plot(df.iloc[:, 0].values, linestyle='--', linewidth=LINEWIDTH, label='OBS')
  plt.plot(df.iloc[:, 1].values, linewidth=LINEWIDTH, label='RAS')
  plt.plot(df.iloc[:, 2].values, linewidth=LINEWIDTH, label='MLP')
  plt.plot(df.iloc[:, 3].values, linewidth=LINEWIDTH, label='RNN')
  plt.plot(df.iloc[:, 4].values, linewidth=LINEWIDTH, label='LSTM')
  plt.plot(df.iloc[:, 5].values, linewidth=LINEWIDTH, label='CNN')
  plt.plot(df.iloc[:, 6].values, linewidth=LINEWIDTH, label='RCNN')

  plt.tick_params(axis='both', which='both', bottom='on', left='on', labelbottom='on', labelleft='on')

  plt.xlabel('Time [hr]', fontsize=24)
  plt.ylabel('Water stage [ft]', fontsize=24)
  plt.xticks(np.arange(0, 24, step=2), [i for i in range(0, 24, 2)], fontsize=22)
  plt.yticks(np.arange(-1, 3, step=0.5), [-1, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=22)
  plt.title('Model response to boundary condition changes on Dec 30, 2020 at S1', fontsize=24)
  plt.legend(fontsize=18, bbox_to_anchor=(1.0, 0.9))

  #plt.savefig('../figures/11-1.png', dpi=400, bbox_inches='tight')
  plt.savefig(outputfile, dpt=400, bbox_inches='tight')
  plt.show()




