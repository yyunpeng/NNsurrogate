# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:05:29 2024

@author: xuyun
"""

#%% libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import TruncatedNormal
from sklearn.preprocessing import MinMaxScaler
import time
import yfinance as yf
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle
from keras.optimizers import Adam
from scipy.optimize import shgo
from statsmodels.tsa.api import VAR
from tensorflow.keras.models import load_model
import os


'''
general NN-approximated solution 的 convergence to the real solution 的数学研究，看在我们这里是不是也可以 apply
我之前写出来的Found a system of equations that the optimal control law satisfies that is yet too difficult to find the closed-form solution，
看我现在做一些numerical experiment能否satisfy that system of equations
'''


#%% global parameters and quantizer set up

T=5
num_processes = 4

# Directory containing the CSV files
directory = '/Users/xuyunpeng/Documents/Time-consistent planning/Meeting19/'

# Initialize an empty list to collect the dataframes
dataframes = []

# Load each CSV file into a dataframe and append to the list
for i in range(1, 6):
    file_path = os.path.join(directory, f'quantizer_vecBrownian/quantization_grid_t_{i}.csv')
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Convert the list of dataframes into a 3D numpy array
number_of_rows = dataframes[0].shape[0]  # Get number of rows from the first dataframe
quantize_grid = np.zeros((T, number_of_rows,  num_processes))

# Fill the numpy array with values from the dataframes
for i, df in enumerate(dataframes):
    quantize_grid[i] = df.values

# Now `quantize_grid` contains the values from the CSV files
print(quantize_grid)

weights = np.full((number_of_rows,), 1/number_of_rows)

start_date, end_date = '2020-01-01', '2023-07-01'

#%% DO NOT RUN: choosing the best 3 stocks

# Define the stock tickers and the date range
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'NFLX', 'PYPL', 'ADBE',
    'INTC', 'CSCO', 'PEP', 'KO', 'DIS', 'V', 'MA', 'JPM', 'BAC', 'WMT',
    'HD', 'PG', 'VZ', 'PFE', 'MRK', 'ABBV', 'T', 'XOM', 'CVX', 'GOVT'
]
start_date = '2021-03-01'
end_date = '2023-03-01'

# Download the adjusted close prices
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']

# Calculate the weekly returns
returns = data.pct_change().dropna()

# Calculate the average return rate and variance of return rate
average_return_rate = returns.mean()  
variance_return_rate = returns.var() 

# Create a DataFrame to hold the average return rate and variance
results = pd.DataFrame({
    "Average Return Rate": average_return_rate,
    "Variance Return Rate": variance_return_rate
})

# Sort by ranked average return rate and ranked variance return rate
sorted_by_avg_return = results.sort_values(by='Average Return Rate', ascending=False)
sorted_by_variance_return = results.sort_values(by='Variance Return Rate', ascending=False)

# Select top 5 stocks with high average and high variance
high_avg_high_var_stock = sorted_by_avg_return.head(10).sort_values(by='Variance Return Rate', ascending=False).head(5)

# Select top 5 stocks with low average and low variance
low_avg_low_var_stock = sorted_by_avg_return.tail(10).sort_values(by='Variance Return Rate', ascending=True).head(10)


#%% write data into csv so R can do MTS, log_claim_differenced are actually 1000*np.random.uniform(0.5, 1.5)

# Seed for reproducibility
# np.random.seed(10)

# Stock tickers
top3 = ['NVDA', 'NFLX', 'GOVT']

# Download data
R = yf.download(top3, start=start_date, end=end_date, interval='1mo', progress=False)['Adj Close']

R = R.pct_change().dropna()
ln_R = np.log(1 + R)
d_ln_R = ln_R.diff().dropna()

# Generate random claims data
variationL = 200
ln_L = np.log(np.random.normal(1000, variationL, size=d_ln_R.shape[0]))
d_ln_L = np.diff(ln_L)
d_ln_L = np.insert(d_ln_L, 0, 0)

# Create DataFrame
toCSV = pd.DataFrame(d_ln_R)
toCSV['claim'] = d_ln_L

# Drop NA values
toCSV = toCSV.dropna()

# Save to CSV
toCSV.to_csv('stock_differences.csv', index=True)
data = pd.read_csv('stock_differences.csv', index_col=0)

# Ensure the index has a monthly frequency
# data = data.asfreq('M')

# Check and convert data types
data = data.apply(pd.to_numeric)

# Fit the VAR model
model = VAR(data)
results = model.fit(maxlags=1, ic='aic')

# Print the results summary
print(results.summary())

phi = results.params

Sigma = results.sigma_u

a = {}
mu = {}
for i in range(phi.shape[0]):
    for j in range(phi.shape[1]):
        if i == 0:
            name = f'{j+1}'
            mu[name] = phi.iloc[i, j]
        else:
            name = f'{i}{j+1}'
            a[name] = phi.iloc[i, j]
            
mu_vec = np.array([mu['1'], mu['2'], mu['3'], mu['4']])

        
#%% Generate test and validation data

numTrain = 1000
numSim = numTrain

def VARMA_sim1(current_dR_temp_list, alpha_matrix_list, Sigma, numSim, T, validation=False):
    
    # if validation:
    #     np.random.seed(10)
    
    noise_test, d_ln_R1_test, d_ln_R2_test, d_ln_R3_test, d_ln_L_test = {}, {}, {}, {}, {}
    
    for t in range(T+3):  # Including 0 for noise
        if t != 0: # For d_R1, d_R2, d_R3, d_L, we start from t=1
            d_ln_R1_test[t], d_ln_R2_test[t], d_ln_R3_test[t], d_ln_L_test[t] = [], [], [], []
        noise_test[t] = []
    
    for path in range(numSim):
        noise = {}
        noise[0] = np.random.normal(0, 1, 4)

        current_dR_temp_list = current_dR_temp_list.copy()  
        
        for t in range(1, T+3):
            noise[t] = np.random.multivariate_normal(mean=np.zeros(4), cov=Sigma)
                                   
            current_dR_temp_list = np.array([a - b  for a, b in zip(current_dR_temp_list, mu_vec)]) 
             
            d_ln_R1 = mu['1'] + np.dot(phi, current_dR_temp_list)[0] + noise[t][0]
            d_ln_R2 = mu['2'] + np.dot(phi, current_dR_temp_list)[1] + noise[t][1]
            d_ln_R3 = mu['3'] + np.dot(phi, current_dR_temp_list)[2] + noise[t][2]
            d_ln_L  = mu['4'] + np.dot(phi, current_dR_temp_list)[3] + noise[t][3]
            
            d_ln_R1_test[t].append((d_ln_R1))
            d_ln_R2_test[t].append((d_ln_R2))
            d_ln_R3_test[t].append((d_ln_R3))
            d_ln_L_test[t].append((d_ln_L))
                        
            current_dR_temp_list = [d_ln_R1, d_ln_R1, d_ln_R1, d_ln_R1]
            
            noise_test[t].append(noise[t])

    # Storing initial noise separately as it does not change with t
    noise_test[0] = [noise[0] for _ in range(numSim)]
    
    return noise_test, d_ln_R1_test, d_ln_R2_test, d_ln_R3_test, d_ln_L_test 

d_ln_R1_0 = d_ln_R[top3[0]][-1] 
d_ln_R2_0 = d_ln_R[top3[1]][-1] 
d_ln_R3_0 = d_ln_R[top3[2]][-1] 
d_ln_L_0 = d_ln_L[-1]

current_R_test = [d_ln_R1_0, d_ln_R2_0, d_ln_R3_0, d_ln_L_0]
noise_vali, d_ln_R1_vali, d_ln_R2_vali, d_ln_R3_vali, d_ln_L_vali = VARMA_sim1( current_R_test, phi, Sigma, numSim, T, validation=True)
noise_test, d_ln_R1_test, d_ln_R2_test, d_ln_R3_test, d_ln_L_test = VARMA_sim1( current_R_test, phi, Sigma, numSim, T)



#%% plot cummulative original scale processes

ln_R1_0 = ln_R[top3[0]][-1]
ln_R2_0 = ln_R[top3[1]][-1]
ln_R3_0 = ln_R[top3[2]][-1]
ln_L_0 = np.log(1000)
# dL_base = 100

time_steps = range(1, T+1)

fig, axs = plt.subplots(4, 1, figsize=(14, 20), sharex=True)

for i in range(numSim):
    axs[0].plot(time_steps,
                [np.exp(ln_R1_0 + sum(d_ln_R1_vali[j][i] for j in range(1, t))) for t in time_steps],
                'b-', alpha=0.2)
    axs[1].plot(time_steps, 
                [np.exp(ln_R2_0 + sum((d_ln_R2_vali[j][i]) for j in range(1, t))) for t in time_steps], 
                'r-', alpha=0.2)
    axs[2].plot(time_steps, 
                [np.exp(ln_R3_0 + sum((d_ln_R3_vali[j][i]) for j in range(1, t))) for t in time_steps], 
                'g-', alpha=0.2)
    axs[3].plot(time_steps, 
                [np.exp(ln_L_0 + sum((d_ln_L_vali[j][i]) for j in range(1, t))) for t in time_steps], 
                'purple', alpha=0.2)


axs[0].set_title('Simulated Paths for R1')
axs[1].set_title('Simulated Paths for R2')
axs[2].set_title('Simulated Paths for R3')
axs[3].set_title('Simulated Paths for L')

for ax in axs:
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True)

plt.tight_layout()
plt.show()

#%% plot VARIMA

time_steps = range(1, T+1)

fig, axs = plt.subplots(4, 1, figsize=(14, 20), sharex=True)

for i in range(numSim):
    axs[0].plot(time_steps, 
                [d_ln_R1_vali[t][i] for t in time_steps], 'b-', alpha=0.2)
    axs[1].plot(time_steps, 
                [d_ln_R2_vali[t][i] for t in time_steps], 'r-', alpha=0.2)
    axs[2].plot(time_steps, 
                [d_ln_R3_vali[t][i] for t in time_steps], 'g-', alpha=0.2)
    axs[3].plot(time_steps, 
                [d_ln_L_vali[t][i] for t in time_steps], 'purple', alpha=0.2)

axs[0].set_title('Simulated Paths for ln_d_R1')
axs[1].set_title('Simulated Paths for ln_d_R2')
axs[2].set_title('Simulated Paths for ln_d_R3')
axs[3].set_title('Simulated Paths for ln_d_L')

for ax in axs:
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True)

plt.tight_layout()
plt.show()


#%% training data 

d_ln_R3_train = np.random.uniform(-0.70, 0.70, numTrain)
d_ln_R2_train = np.random.uniform(-0.80, 0.80, numTrain)
d_ln_R1_train = np.random.uniform(-0.06, 0.06, numTrain)
d_ln_L_train = np.random.uniform(-0.65, -0.80, numTrain)

R3_t_train = np.random.uniform(0.40, 3.20, numTrain)
R2_t_train = np.random.uniform(0.50, 3.00, numTrain)
R1_t_train = np.random.uniform(1.00, 1.25, numTrain)
L_t_train = np.random.uniform(400, 3500, numTrain) 

y = 1200
gamma = 0.8
v = 0.9

c_min = 100000-1200*(T-1)
c_max = 100000+1200*(T-1)
y_train = np.ones(numTrain) * y
c_train = np.random.uniform(c_min, c_max, numTrain)

# noise_R1_train = np.random.uniform(-5, 5, numTrain)
# noise_R2_train = np.random.uniform(-5, 5, numTrain)
# noise_R3_train = np.random.uniform(-5, 5, numTrain)
# noise_L_train = np.random.uniform(-5, 5, numTrain)

gamma_train = np.ones(numTrain) * gamma
v_train = np.ones(numTrain) * v

a11_train, a12_train, a13_train, a14_train = np.ones(numTrain) * a['11'], np.ones(numTrain) * a['12'], np.ones(numTrain) * a['13'], np.ones(numTrain) * a['14']
a21_train, a22_train, a23_train, a24_train = np.ones(numTrain) * a['21'], np.ones(numTrain) * a['22'], np.ones(numTrain) * a['23'], np.ones(numTrain) * a['24']
a31_train, a32_train, a33_train, a34_train = np.ones(numTrain) * a['31'], np.ones(numTrain) * a['32'], np.ones(numTrain) * a['33'], np.ones(numTrain) * a['34']
a41_train, a42_train, a43_train, a44_train = np.ones(numTrain) * a['41'], np.ones(numTrain) * a['42'], np.ones(numTrain) * a['43'], np.ones(numTrain) * a['44']

mu1_train, mu2_train, mu3_train, muL_train = np.ones(numTrain) * mu['1'], np.ones(numTrain) * mu['2'], np.ones(numTrain) * mu['3'], np.ones(numTrain) * mu['4']
# dL_base_train = np.ones(numTrain) * 100

#%% terminal value function set up

def V_T(gamma, C_T):
    return 1/gamma * np.sign(C_T) * (np.abs(C_T)) ** gamma

def E_V_T(c,y1,gamma,v, u , 
            d_ln_R1t, d_ln_R2t, d_ln_R3t, d_ln_Lt,
            R1t, R2t, R3t, Lt, 
            # noise_R1, noise_R2, noise_R3, noise_L,
              a11, a12, a13, a14,
              a21, a22, a23, a24,
              a31, a32, a33, a34,
              a41, a42, a43, a44,
              # b11, b12, b13, b14,
              # b21, b22, b23, b24,
              # b31, b32, b33, b34,
              # b41, b42, b43, b44,
              mu1, mu2, mu3, muL,
              # dL_base,
            quantizer, t=T-1):
    
    matrix_alpha = np.array([
        [a11, a12, a13, a14], [a21, a22, a23, a24], [a31, a32, a33, a34], [a41, a42, a43, a44]
        ])
    # matrix_beta = [b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34, b41, b42, b43, b44]
    mu_vec = np.array([mu1, mu2, mu3, muL])
    d_ln_Rt_vec = np.array([d_ln_R1t - mu1, d_ln_R2t - mu2, d_ln_R3t - mu3, d_ln_Lt - muL])
    # noise_t_vec = np.array([noise_R1, noise_R2, noise_R3, noise_L])    

    r1 = np.exp(
        np.log(R1t) + (mu_vec[0] + np.dot(matrix_alpha, d_ln_Rt_vec)[0] #+ dotProduc(matrix_beta,noise_t_vec)[0]
         + quantizer[t][:, 0])
        )
    r2 = np.exp(
        np.log(R2t) + (mu_vec[1] + np.dot(matrix_alpha, d_ln_Rt_vec)[1] #+ dotProduc(matrix_beta,noise_t_vec)[1]
         + quantizer[t][:, 1])
        )
    r3 = np.exp(
        np.log(R3t) + (mu_vec[2] + np.dot(matrix_alpha, d_ln_Rt_vec)[2] #+ dotProduc(matrix_beta,noise_t_vec)[2]
         + quantizer[t][:, 2])
        )
    L = np.exp(
        np.log(Lt) + (mu_vec[3] + np.dot(matrix_alpha, d_ln_Rt_vec)[3] #+ dotProduc(matrix_beta,noise_t_vec)[3]
         + quantizer[t][:, 3])
        )
    
    c_next = (u[0]*r1 + u[1]*r2 + u[2]*r3)*(c + y1- u[3]*c) - L
    
    vec = 1/gamma * np.sign(u[3]*c) * (np.abs(u[3]*c)) ** gamma + v*V_T(gamma, c_next)
    
    return np.sum(vec*weights)


#%% NN Predictor set up

def NN_Surrogate(c1,y1,gamma,v,
              d_ln_R1t, d_ln_R2t, d_ln_R3t, d_ln_Lt,
              R1t, R2t, R3t, Lt,
              # noise_R1, noise_R2, noise_R3, noise_L, 
                a11, a12, a13, a14,
                a21, a22, a23, a24,
                a31, a32, a33, a34,
                a41, a42, a43, a44,
                # b11, b12, b13, b14,
                # b21, b22, b23, b24,
                # b31, b32, b33, b34,
                # b41, b42, b43, b44,
                mu1, mu2, mu3, muL, 
                # dL_base,
              nnweights, inputscaler, outputscaler, scaleOutput = 1):
    
    inputdata = np.concatenate((
                                c1.reshape(-1,1),
                                y1.reshape(-1,1),
                                gamma.reshape(-1,1), 
                                v.reshape(-1,1),
                                
                                d_ln_R1t.reshape(-1,1), 
                                d_ln_R2t.reshape(-1,1), 
                                d_ln_R3t.reshape(-1,1), 
                                d_ln_Lt.reshape(-1,1), 
                                                                                                
                                R1t.reshape(-1,1), 
                                R2t.reshape(-1,1), 
                                R3t.reshape(-1,1), 
                                Lt.reshape(-1,1),
                                
                                # noise_R1.reshape(-1,1), 
                                # noise_R2.reshape(-1,1), 
                                # noise_R3.reshape(-1,1), 
                                # noise_L.reshape(-1,1), 
                                
                                a11.reshape(-1,1), a12.reshape(-1,1), a13.reshape(-1,1), a14.reshape(-1,1),
                                a21.reshape(-1,1), a22.reshape(-1,1), a23.reshape(-1,1), a24.reshape(-1,1),
                                a31.reshape(-1,1), a32.reshape(-1,1), a33.reshape(-1,1), a34.reshape(-1,1),
                                a41.reshape(-1,1), a42.reshape(-1,1), a43.reshape(-1,1), a44.reshape(-1,1),
                                # b11.reshape(-1,1), b12.reshape(-1,1), b13.reshape(-1,1), b14.reshape(-1,1),
                                # b21.reshape(-1,1), b22.reshape(-1,1), b23.reshape(-1,1), b24.reshape(-1,1),
                                # b31.reshape(-1,1), b32.reshape(-1,1), b33.reshape(-1,1), b34.reshape(-1,1),
                                # b41.reshape(-1,1), b42.reshape(-1,1), b43.reshape(-1,1), b44.reshape(-1,1),
                                mu1.reshape(-1,1), mu2.reshape(-1,1), mu3.reshape(-1,1), muL.reshape(-1,1),
                                # dL_base.reshape(-1,1)
                         
                                ), axis = 1)
    
    inputdata = inputscaler.transform(inputdata)
    
    layer1out = np.dot(inputdata, nnweights[0]) + nnweights[1]
    
    layer1out = tf.keras.activations.elu(layer1out).numpy()
    
    layer2out = np.dot(layer1out, nnweights[2]) + nnweights[3]
    
    layer2out = tf.keras.activations.elu(layer2out).numpy()
    
    layer3out = np.dot(layer2out, nnweights[4]) + nnweights[5]
    
    layer3out = tf.keras.activations.elu(layer3out).numpy()
    
    layer4out = np.dot(layer3out, nnweights[6]) + nnweights[7]
    
    layer4out = tf.keras.activations.elu(layer4out).numpy()
    
    layer5out = np.dot(layer4out, nnweights[8]) + nnweights[9]
    
    layer5out = tf.keras.activations.elu(layer5out).numpy()
    
    layer6out = np.dot(layer5out, nnweights[10]) + nnweights[11]
    
    if scaleOutput == 0:   # for pu apply softmax
        output = tf.keras.activations.sigmoid(layer6out).numpy() 
    if scaleOutput == 1:   # for value function apply output scaler
        output = outputscaler.inverse_transform(layer6out)
    
    return output


#%% value function surrogate set up

def E_V_t_plus1(c1,y1,gamma,v, u , 
                d_ln_R1t, d_ln_R2t, d_ln_R3t, d_ln_Lt,
                R1t, R2t, R3t, Lt,
                # noise_R1, noise_R2, noise_R3, noise_L,
                  a11, a12, a13, a14,
                  a21, a22, a23, a24,
                  a31, a32, a33, a34,
                  a41, a42, a43, a44,
                  # b11, b12, b13, b14,
                  # b21, b22, b23, b24,
                  # b31, b32, b33, b34,
                  # b41, b42, b43, b44,
                  mu1, mu2, mu3, muL,
                  # dL_base,
                nnweights, inputscaler, outputscaler, quantizer, t):
    
    numWeights = len(quantizer[0])
    
    matrix_alpha = np.array([
        [a11, a12, a13, a14], [a21, a22, a23, a24], [a31, a32, a33, a34], [a41, a42, a43, a44]
        ])
    # matrix_beta = [b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34, b41, b42, b43, b44]
    mu_vec = np.array([mu1, mu2, mu3, muL])
    d_ln_Rt_vec = np.array([d_ln_R1t - mu1, d_ln_R2t - mu2, d_ln_R3t - mu3, d_ln_Lt - muL])
    # noise_t_vec = np.array([noise_R1, noise_R2, noise_R3, noise_L])    

    r1 = np.exp(
        np.log(R1t) + (mu_vec[0] + np.dot(matrix_alpha, d_ln_Rt_vec)[0] #+ dotProduc(matrix_beta,noise_t_vec)[0]
         + quantizer[t][:, 0])
        )
    r2 = np.exp(
        np.log(R2t) + (mu_vec[1] + np.dot(matrix_alpha, d_ln_Rt_vec)[1] #+ dotProduc(matrix_beta,noise_t_vec)[1]
         + quantizer[t][:, 1])
        )
    r3 = np.exp(
        np.log(R3t) + (mu_vec[2] + np.dot(matrix_alpha, d_ln_Rt_vec)[2] #+ dotProduc(matrix_beta,noise_t_vec)[2]
         + quantizer[t][:, 2])
        )
    L = np.exp(
        np.log(Lt) + (mu_vec[3] + np.dot(matrix_alpha, d_ln_Rt_vec)[3] #+ dotProduc(matrix_beta,noise_t_vec)[3]
         + quantizer[t][:, 3])
        )
    
    temp = 1/gamma * np.sign(u[3]*c1) * (np.abs(u[3]*c1)) ** gamma \
            + v* NN_Surrogate(np.ones(numWeights) * (u[0]*r1 + u[1]*r2 + u[2]*r3)*(c1 + y1- u[3]*c1) - L,
                               np.ones(numWeights) * y1,
                               np.ones(numWeights) * gamma,
                               np.ones(numWeights) * v,
                               
                               np.ones(numWeights) * d_ln_R1t, 
                               np.ones(numWeights) * d_ln_R2t, 
                               np.ones(numWeights) * d_ln_R3t, 
                               np.ones(numWeights) * d_ln_Lt, 
    
                               np.ones(numWeights) * R1t, 
                               np.ones(numWeights) * R2t, 
                               np.ones(numWeights) * R3t, 
                               np.ones(numWeights) * Lt, 
                                                          
                               # np.ones(numWeights) * noise_R1, 
                               # np.ones(numWeights) * noise_R2, 
                               # np.ones(numWeights) * noise_R3, 
                               # np.ones(numWeights) * noise_L, 
                               
                                np.ones(numWeights) * a11,np.ones(numWeights) * a12,np.ones(numWeights) * a13,np.ones(numWeights) * a14,
                                np.ones(numWeights) * a21,np.ones(numWeights) * a22,np.ones(numWeights) * a23,np.ones(numWeights) * a24,
                                np.ones(numWeights) * a31,np.ones(numWeights) * a32,np.ones(numWeights) * a33,np.ones(numWeights) * a34,
                                np.ones(numWeights) * a41,np.ones(numWeights) * a42,np.ones(numWeights) * a43,np.ones(numWeights) * a44,
                                # np.ones(numWeights) * b11,np.ones(numWeights) * b12,np.ones(numWeights) * b13,np.ones(numWeights) * b14,
                                # np.ones(numWeights) * b21,np.ones(numWeights) * b22,np.ones(numWeights) * b23,np.ones(numWeights) * b24,
                                # np.ones(numWeights) * b31,np.ones(numWeights) * b32,np.ones(numWeights) * b33,np.ones(numWeights) * b34,
                                # np.ones(numWeights) * b41,np.ones(numWeights) * b42,np.ones(numWeights) * b43,np.ones(numWeights) * b44,
                                np.ones(numWeights) * mu1,np.ones(numWeights) * mu2,np.ones(numWeights) * mu3,np.ones(numWeights) * muL,
                                # np.ones(numWeights) * dL_base,

                           nnweights, inputscaler, outputscaler)
            
    v_m_n = np.sum(temp.flatten() * weights)
    
    return v_m_n

#%% training setup 

# def scale(x, new_min=0.5, new_max=1.5):
#     # Use a sigmoid function to squash the input to the range [0, 1]
#     normalized_x = 1 / (1 + np.exp(-x))
#     # Scale and shift the normalized value to the new range
#     mapped_value = new_min + normalized_x * (new_max - new_min)
#     return mapped_value
def custom_activation(x):
    return tf.nn.sigmoid(x)*0.001

#%% train

def BuildAndTrainModel(c1_train, y1_train, gamma_train, v_train, 
                        d_ln_R1_train, d_ln_R2_train, d_ln_R3_train, d_ln_L_train,
                        R1_t_train, R2_t_train, R3_t_train, L_t_train, 
                        # noise_R1_train, noise_R2_train, noise_R3_train, noise_L_train, 
                        a11_train,a12_train,a13_train,a14_train,
                        a21_train,a22_train,a23_train,a24_train,
                        a31_train,a32_train,a33_train,a34_train,
                        a41_train,a42_train,a43_train,a44_train,
                        # b11_train,b12_train,b13_train,b14_train,
                        # b21_train,b22_train,b23_train,b24_train,
                        # b31_train,b32_train,b33_train,b34_train,
                        # b41_train,b42_train,b43_train,b44_train,
                        mu1_train,mu2_train,mu3_train,muL_train,
                        # dL_base_train,

                       quantizer, 
                       nn_dim = 32, 
                       
                       node_num = 150, 
                       batch_num = 60, 
                       epoch_num = 5000, 
                       initializer = TruncatedNormal(mean = 0.0, stddev = 0.05, seed = 0) 
                       
                       ):
        
    input_train = np.concatenate((
                                    c1_train.reshape(-1,1),
                                    y1_train.reshape(-1,1),
                                    gamma_train.reshape(-1,1), 
                                    v_train.reshape(-1,1),
                                    
                                    d_ln_R1_train.reshape(-1,1), 
                                    d_ln_R2_train.reshape(-1,1), 
                                    d_ln_R3_train.reshape(-1,1), 
                                    d_ln_L_train.reshape(-1,1), 
                                    
                                    R1_t_train.reshape(-1,1), 
                                    R2_t_train.reshape(-1,1), 
                                    R3_t_train.reshape(-1,1), 
                                    L_t_train.reshape(-1,1),
                                    
                                    # noise_R1_train.reshape(-1,1),
                                    # noise_R2_train.reshape(-1,1),
                                    # noise_R3_train.reshape(-1,1),
                                    # noise_L_train.reshape(-1,1),
                                    
                                    a11_train.reshape(-1,1),a12_train.reshape(-1,1),a13_train.reshape(-1,1),a14_train.reshape(-1,1),
                                    a21_train.reshape(-1,1),a22_train.reshape(-1,1),a23_train.reshape(-1,1),a24_train.reshape(-1,1),
                                    a31_train.reshape(-1,1),a32_train.reshape(-1,1),a33_train.reshape(-1,1),a34_train.reshape(-1,1),
                                    a41_train.reshape(-1,1),a42_train.reshape(-1,1),a43_train.reshape(-1,1),a44_train.reshape(-1,1),
                                    # b11_train.reshape(-1,1),b12_train.reshape(-1,1),b13_train.reshape(-1,1),b14_train.reshape(-1,1),
                                    # b21_train.reshape(-1,1),b22_train.reshape(-1,1),b23_train.reshape(-1,1),b24_train.reshape(-1,1),
                                    # b31_train.reshape(-1,1),b32_train.reshape(-1,1),b33_train.reshape(-1,1),b34_train.reshape(-1,1),
                                    # b41_train.reshape(-1,1),b42_train.reshape(-1,1),b43_train.reshape(-1,1),b44_train.reshape(-1,1),
                                    mu1_train.reshape(-1,1),mu2_train.reshape(-1,1),mu3_train.reshape(-1,1),muL_train.reshape(-1,1)
                                    # dL_base_train.reshape(-1,1)

                                    ), axis = 1) 
    
    
    input_scaler = MinMaxScaler(feature_range = (0,1))
    input_scaler.fit(input_train)
    input_train_scaled = input_scaler.transform(input_train)
    
    valuefun_train = np.zeros((T+1, numTrain))
    proportion_train = np.zeros((3, T+1, numTrain))
    dividend_train = np.zeros((T+1, numTrain))
    
    output_scaler_valuefun = np.empty(T+1, dtype = object)
    nnsolver_valuefun = np.empty(T+1, dtype = object)
    nnsolver_proportion = np.empty(T+1, dtype=object)
    nnsolver_dividend = np.empty(T+1, dtype=object)
    
    check_minimise_1000 = {}
    
    start = time.perf_counter() 
    
    # Run through all time steps backwards 
    for j in range(T-1, 0, -1): # j is equivalent to t
    
        check_minimise_1000[j] = {}
        
        start_i = time.perf_counter()
        print("Time step " + str(j))
        

        # Create training output for value function and policy        
        for i in range(numTrain):
            
            check_minimise_1000[j][i] = {}
            
            if j < (T-1):
                output_scaler = output_scaler_valuefun[j+1]  
                
                def f_i(u):
                        V = E_V_t_plus1(
                                                    c1_train[i], y1_train[i], gamma_train[i], v_train[i], u,
                                                    d_ln_R1_train[i], d_ln_R2_train[i], d_ln_R3_train[i], d_ln_L_train[i],
                                                    R1_t_train[i], R2_t_train[i], R3_t_train[i], L_t_train[i],
                                                    # noise_R1_train[i], noise_R2_train[i], noise_R3_train[i], noise_L_train[i],
                                                    a11_train[i], a12_train[i], a13_train[i], a14_train[i],
                                                    a21_train[i], a22_train[i], a23_train[i], a24_train[i],
                                                    a31_train[i], a32_train[i], a33_train[i], a34_train[i],
                                                    a41_train[i], a42_train[i], a43_train[i], a44_train[i],
                                                    # b11_train[i], b12_train[i], b13_train[i], b14_train[i],
                                                    # b21_train[i], b22_train[i], b23_train[i], b24_train[i],
                                                    # b31_train[i], b32_train[i], b33_train[i], b34_train[i],
                                                    # b41_train[i], b42_train[i], b43_train[i], b44_train[i],
                                                    mu1_train[i], mu2_train[i], mu3_train[i], muL_train[i],
                                                    # dL_base_train[i],
                                                    nnsolver_valuefun[j+1].get_weights(),
                                                    input_scaler, output_scaler, quantizer, j
                                                )
                        return -1*V + np.abs(np.sum(u) - 1)*50000

                            
# this output scaler valufun is where DP is incorporated, every previous period optimizaton takes the numeric 
# value of the last value function

            else:

                def f_i(u):
                        V = E_V_T(
                                                    c1_train[i], y1_train[i], gamma_train[i], v_train[i], u,
                                                    d_ln_R1_train[i], d_ln_R2_train[i], d_ln_R3_train[i], d_ln_L_train[i],
                                                    R1_t_train[i], R2_t_train[i], R3_t_train[i], L_t_train[i],
                                                    # noise_R1_train[i], noise_R2_train[i], noise_R3_train[i], noise_L_train[i],
                                                    a11_train[i], a12_train[i], a13_train[i], a14_train[i],
                                                    a21_train[i], a22_train[i], a23_train[i], a24_train[i],
                                                    a31_train[i], a32_train[i], a33_train[i], a34_train[i],
                                                    a41_train[i], a42_train[i], a43_train[i], a44_train[i],
                                                    # b11_train[i], b12_train[i], b13_train[i], b14_train[i],
                                                    # b21_train[i], b22_train[i], b23_train[i], b24_train[i],
                                                    # b31_train[i], b32_train[i], b33_train[i], b34_train[i],
                                                    # b41_train[i], b42_train[i], b43_train[i], b44_train[i],
                                                    mu1_train[i], mu2_train[i], mu3_train[i], muL_train[i],
                                                    # dL_base_train[i],
                                                    quantizer
                                                    )                        
                        return -1*V + np.abs(np.sum(u) - 1)*50000

            div_upper = 0.05
            bounds = [(0, 1), (0, 1), (0, 1), (0, div_upper)]
            def constraint_sum(x):
                return x[0]+x[1]+x[2] - 1
            constraints = [
                {'type': 'eq', 'fun': constraint_sum}
            ]
            # minimizer_kwargs = {
            #     'method': 'Powell',  # Local optimizer method
            #     'bounds': bounds       # Bounds for the local optimizer
            # }
            result_global = shgo(f_i, bounds, constraints=constraints, 
                                   # minimizer_kwargs=minimizer_kwargs, 
                                    # n=10, iters=10, 
                                    # options={'maxiter': 100}, 
                                    # sampling_method='sobol'
                                  )
            # print(result_global.x)
            if i == int(numTrain/2) or i == int(numTrain/4*3) or i == int(numTrain/4):
                print(f'          {i}th optimization done')
            
            for k in range(3):    
                proportion_train[k][j][i] = result_global.x[k] 
            dividend_train[j][i] = result_global.x[-1] 
            valuefun_train[j][i] = result_global.fun*-1
            
            check_minimise_1000[j][i]['proportion'] = proportion_train[:,j,i]
            check_minimise_1000[j][i]['dividend'] = dividend_train[j][i]
            check_minimise_1000[j][i]['valuefun'] = valuefun_train[j][i]

        
        end_i = time.perf_counter()
        print("     all optimizations done: " + str(round((end_i-start_i)/60,2)) + " min.")
        
        start_i = time.perf_counter()
        output_scaler_valuefun[j] = MinMaxScaler(feature_range = (0,1))
        output_scaler_valuefun[j].fit(valuefun_train[j].reshape(-1, 1))
        valuefun_train_scaled = output_scaler_valuefun[j].transform(valuefun_train[j].reshape(-1,1))    
        nnsolver_valuefun[j] = Sequential([
                                        Input(shape=(nn_dim,)),  # Explicit input layer specification
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(1, activation=None, kernel_initializer=initializer, bias_initializer=initializer)
                                        ])
        optimizer = Adam(learning_rate=0.0001)
        nnsolver_valuefun[j].compile(optimizer = optimizer, loss = 'mean_absolute_error')
        nnsolver_valuefun[j].fit(input_train_scaled, valuefun_train_scaled,
                              epochs = epoch_num, batch_size = batch_num, verbose = 0)
        end_i = time.perf_counter()
        print("     train value function done: " + str(round((end_i-start_i)/60,2)) + " min.")       
        
        start_i = time.perf_counter()        
        nnsolver_proportion[j] = Sequential([
                                    Input(shape=(nn_dim,)),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(3, activation='softmax', kernel_initializer=initializer, bias_initializer=initializer)
                                    ])
        optimizer = Adam(learning_rate=0.0001)
        nnsolver_proportion[j].compile(optimizer=optimizer, loss='mean_absolute_error')
        nnsolver_proportion[j].fit(input_train_scaled, proportion_train[:, j, :].T,
                               epochs=epoch_num, batch_size=batch_num, verbose=0) 
        end_i = time.perf_counter()
        print("     train proportion done: " + str(round((end_i-start_i)/60,2)) + " min.")
        
        start_i = time.perf_counter()   

        nnsolver_dividend[j] = Sequential([
                                    Input(shape=(nn_dim,)),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(1, activation=custom_activation, kernel_initializer=initializer, bias_initializer=initializer)
                                    ])
        optimizer = Adam(learning_rate=0.0001)
        nnsolver_dividend[j].compile(optimizer=optimizer, loss='mean_absolute_error')
        nnsolver_dividend[j].fit(input_train_scaled, dividend_train[j].reshape(-1, 1),
                               epochs=epoch_num, batch_size=batch_num, verbose=0) 
        end_i = time.perf_counter()
        print("     train dividend done: " + str(round((end_i-start_i)/60,2)) + " min.")
        
    end = time.perf_counter()
    duration = (end-start)/60

    print("Duration: " + str(duration) + " min.")
    
    return nnsolver_proportion, nnsolver_dividend, nnsolver_valuefun, input_scaler, output_scaler_valuefun, check_minimise_1000

'print出来mean square error，把这些expected utility都存下来'
'做validation：做一个validation data set（可以是simulation），把由不同的hyper parameter的train出来的model放进去（相当于现在的test的步骤），比较expected utility'
'再用由最大expected utility的validation的hyper parameter的model去test' 
'我们假设simulation的data是真正的不知道的未来的data。我们假设我们有一个对未来的预估，即，validation dataset，用最好的validation的hyper parameter去test。'

#%% Train

nnsolver_proportion, nnsolver_dividend, nnsolver_valuefun, in_scaler, out_scaler_valuefun, check_minimise_1000 \
= BuildAndTrainModel(c_train, y_train, gamma_train, v_train, 
                        d_ln_R1_train, d_ln_R2_train, d_ln_R3_train, d_ln_L_train,
                        R1_t_train, R2_t_train, R3_t_train, L_t_train, 
                        # noise_R1_train, noise_R2_train, noise_R3_train, noise_L_train, 
                        a11_train,a12_train,a13_train,a14_train,
                        a21_train,a22_train,a23_train,a24_train,
                        a31_train,a32_train,a33_train,a34_train,
                        a41_train,a42_train,a43_train,a44_train,
                        # b11_train,b12_train,b13_train,b14_train,
                        # b21_train,b22_train,b23_train,b24_train,
                        # b31_train,b32_train,b33_train,b34_train,
                        # b41_train,b42_train,b43_train,b44_train,
                        mu1_train,mu2_train,mu3_train,muL_train,
                        # dL_base_train,
                     quantize_grid)

#%% save locally

np.save('/Users/xuyunpeng/Documents/Time-consistent planning/Meeting19/models/nnsolver_proportion_19-3', nnsolver_proportion)
np.save('/Users/xuyunpeng/Documents/Time-consistent planning/Meeting19/models/nnsolver_valuefun_19-3', nnsolver_valuefun)
np.save('/Users/xuyunpeng/Documents/Time-consistent planning/Meeting19/models/out_scaler_valuefun_19-3', out_scaler_valuefun)

for j in range(1,T):
    nnsolver_dividend[j].save(f'nnsolver_dividend_19-3_t{j}.keras')

file_path = '/Users/xuyunpeng/Documents/Time-consistent planning/Meeting19/models/in_scaler_19-3.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(in_scaler, file)


#%% retrieve models

print('remember to change working directory, otherwise File not found: filepath=nnsolver_dividend_19-1_t1.keras.')

loaded_proportion = np.load('/Users/xuyunpeng/Documents/Time-consistent planning/Meeting19/models/nnsolver_proportion_19-2.npy', allow_pickle=True)

loaded_dividend = []
for j in range(1,T):
    loaded_dividend.append(load_model(f'nnsolver_dividend_19-2_t{j}.keras', custom_objects={'custom_activation': custom_activation}))
loaded_dividend = [None] + loaded_dividend + [None]
    
file_path = '/Users/xuyunpeng/Documents/Time-consistent planning/Meeting19/models/in_scaler_19-2.pkl'
with open(file_path, 'rb') as file:
    loaded_in_scaler = pickle.load(file)
    

#%% test one path

def IndividualTest(c0, gamma, v, y, nnsolver_proportion, nnsolver_dividend, path, input_scaler, T,
                    a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44, 
                    # b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34, b41, b42, b43, b44, 
                    mu1, mu2, mu3, muL, 
                    d_ln_R1, d_ln_R2, d_ln_R3, d_ln_L
                   ):
    
    samples = np.ones((1, 5, T+2))  
                                                   
    samples[:,4,0:T+2] = c0
    
    for t in range(1, T):
                
        if t < (T):
            
            u_NN = np.empty(3)
            
            for i in range(3):
            
                # NN strategy 
                u_NN[i] = NN_Surrogate(samples[0][4][t], y, gamma, v, 
                                             
                                             d_ln_R1[t][path], 
                                             d_ln_R2[t][path], 
                                             d_ln_R3[t][path], 
                                             d_ln_L[t][path],                                                  
                                             
                                             np.array(
                                                 np.exp((ln_R1_0) + sum((d_ln_R1[j][path]) for j in range(1, t)))
                                                   ), 
                                             np.array(
                                                 np.exp((ln_R2_0) + sum((d_ln_R2[j][path]) for j in range(1, t)))
                                                   ), 
                                             np.array(
                                                 np.exp((ln_R3_0) + sum((d_ln_R3[j][path]) for j in range(1, t)))
                                                   ), 
                                             np.array(
                                                 np.exp((ln_L_0) + sum((d_ln_L[j][path]) for j in range(1, t)))
                                                   ), 
                                                                                              
                                             # noise_test[t][path][0], 
                                             # noise_test[t][path][1],
                                             # noise_test[t][path][2],
                                             # noise_test[t][path][3],
                                             
                                             a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44, 
                                             # b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34, b41, b42, b43, b44, 
                                             mu1, mu2, mu3, muL, 
                                             
                                             nnsolver_proportion[t].get_weights(), input_scaler, 
                                             outputscaler=None, scaleOutput=0)[0,i]
                
            # make sure they are proportions and sum up to 1
            samples[0][0][t] = u_NN[0]/sum(u_NN[i] for i in range(3))
            samples[0][1][t] = u_NN[1]/sum(u_NN[i] for i in range(3))
            samples[0][2][t] = u_NN[2]/sum(u_NN[i] for i in range(3))
                
            # for nnsolver_dividend, the index starts from 0, to 3. 
            # j = t-1
            samples[0][3][t] = NN_Surrogate(samples[0][4][t], y, gamma, v, 
                                             
                                              d_ln_R1[t][path], 
                                              d_ln_R2[t][path], 
                                              d_ln_R3[t][path], 
                                              d_ln_L[t][path],                                                    
                                              
                                              np.array(
                                                  np.exp((ln_R1_0) + sum((d_ln_R1[j][path]) for j in range(1, t)))
                                                    ), 
                                              np.array(
                                                  np.exp((ln_R2_0) + sum((d_ln_R2[j][path]) for j in range(1, t)))
                                                    ), 
                                              np.array(
                                                  np.exp((ln_R3_0) + sum((d_ln_R3[j][path]) for j in range(1, t)))
                                                    ), 
                                              np.array(
                                                  np.exp((ln_L_0) + sum((d_ln_L[j][path]) for j in range(1, t)))
                                                    ), 
                                                                                              
                                              # noise_test[t][path][0], 
                                              # noise_test[t][path][1],
                                              # noise_test[t][path][2],
                                              # noise_test[t][path][3],
                                             
                                              a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44, 
                                              # b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34, b41, b42, b43, b44, 
                                              mu1, mu2, mu3, muL, 
                                             
                                              nnsolver_dividend[j].get_weights(), input_scaler, 
                                              outputscaler=None, scaleOutput=0)[0][0]
                

            
            # counter = 0 # to see how many times the sum of control is too far away from 1 
            # if sum(u_NN[i] for i in range(3)) >1.1 or sum(u_NN[i] for i in range(3)) < 0.9:
            #     counter+=1
            
            samples[0][4][t+1] = \
            (samples[0][0][t] * np.exp((ln_R1_0) + sum((d_ln_R1[j][path]) for j in range(1, t+1))) +
             samples[0][1][t] * np.exp((ln_R2_0) + sum((d_ln_R2[j][path]) for j in range(1, t+1))) +
             samples[0][2][t] * np.exp((ln_R3_0) + sum((d_ln_R3[j][path]) for j in range(1, t+1))) ) *\
                (samples[0][4][t] + y - samples[0][4][t]*samples[0][3][t] ) - \
                    np.exp(ln_L_0) + sum((d_ln_L[j][path]) for j in range(1, t+1)) 
    
    loss_count = 0
    minimum_capital = c0 - sum(
                            np.exp(ln_L_0 + sum(d_ln_L[j][path] for j in range(1, t))) 
                            for t in range(1, T)
                            ) + (T-1)*y
    if samples[0][4][T-1] < minimum_capital:
        loss_count = 1
        
    return samples, loss_count


def RunTests(c0, gamma, v, y, nnsolver_proportion, nnsolver_dividend, input_scaler, T, numSim,
                    a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44, 
                    # b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34, b41, b42, b43, b44, 
                    mu1, mu2, mu3, muL, 
                    d_R1, d_R2, d_R3, d_L
                    ):    
    
    results = {}
    total_loss_coun = 0

    for path in range(1,numSim):
        
        samples, loss_coun = IndividualTest(c0, gamma, v, y, nnsolver_proportion, nnsolver_dividend, path, input_scaler, T,
                    a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44, 
                    # b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34, b41, b42, b43, b44, 
                    mu1, mu2, mu3, muL, 
                    d_R1, d_R2, d_R3, d_L
                   )
        
        results[path] = samples
        total_loss_coun += loss_coun 

    return results, total_loss_coun


#%% Validation results

initial_capital = 100000

results_vali, total_loss_count_vali = RunTests(initial_capital, np.array(gamma), np.array(v), np.array(y), 
                                     
                    loaded_proportion, loaded_dividend, loaded_in_scaler, 
                    # nnsolver_proportion, nnsolver_dividend, in_scaler,
                   
                   T, numSim,
                    np.array(a['11']), np.array(a['12']), np.array(a['13']), np.array(a['14']), 
                    np.array(a['21']), np.array(a['22']), np.array(a['23']), np.array(a['24']), 
                    np.array(a['31']), np.array(a['32']), np.array(a['33']), np.array(a['34']), 
                    np.array(a['41']), np.array(a['42']), np.array(a['43']), np.array(a['44']), 
                    np.array(mu['1']), np.array(mu['2']), np.array(mu['3']), np.array(mu['4']), 
                    # np.array(dL_base),
                    
                    d_ln_R1_vali, d_ln_R2_vali, d_ln_R3_vali, d_ln_L_vali
                   )

capit_NN = []
for path in results_vali:
    capit_NN.append(results_vali[path][0][4][T])

mean_NN = round(np.mean(capit_NN),3)
percentiles = [25, 50, 75]
NN_percentile = np.percentile(capit_NN, percentiles)


print('NeuN mean =', mean_NN)

print('percentile =', round(NN_percentile[0],3), round(NN_percentile[1],3), round(NN_percentile[2],3))

print('Pr(NN loss money)=',total_loss_count_vali/numSim)

print('average return rate monthly = ', (mean_NN/initial_capital)**(1/(T-1))-1)
# print('percentage: number of times the sum of control is too far away from 1 is ', total_counter/numSim)


#%% print some results in detail

'check why liabilities are always around 1000'

some_paths = [random.randint(1, numSim-1) for _ in range(5)]

for some in range(1,len(some_paths)):
    print('')
    print(f'for tested path no.{some}')
    for t in range(1,T):
        print(
             f'u0_{t}=', round(results_vali[some_paths[some]][0][0][t],4),
             f'u1_{t}=', round(results_vali[some_paths[some]][0][1][t],4),
             f'u2_{t}=', round(results_vali[some_paths[some]][0][2][t],4),
             f'u3_{t}=', round(results_vali[some_paths[some]][0][3][t],4)
              )
        print(
              f'                                                     L_{t} =', round(np.exp((ln_L_0) + sum((d_ln_L_vali[j][some]) for j in range(1, t))),2), f'C_{t} =', round(results_vali[some_paths[some]][0][4][t],2)
              
              )

#%%

import matplotlib.pyplot as plt

import random

bottomLine = 100000 + (T-1) * 1200

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(capit_NN, bins=80, color='blue', edgecolor='black')
plt.axvline(x=bottomLine, color='red', linestyle='--', linewidth=2, label=initial_capital)

plt.title('Histogram of capit_NN')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


#%% Test results

initial_capital = 100000

results_test, total_loss_count_test = RunTests(initial_capital, np.array(gamma), np.array(v), np.array(y), 
                                     
                    loaded_proportion, loaded_dividend, loaded_in_scaler, 
                    # nnsolver_proportion, nnsolver_dividend, in_scaler,
                   
                   T, numSim,
                    np.array(a['11']), np.array(a['12']), np.array(a['13']), np.array(a['14']), 
                    np.array(a['21']), np.array(a['22']), np.array(a['23']), np.array(a['24']), 
                    np.array(a['31']), np.array(a['32']), np.array(a['33']), np.array(a['34']), 
                    np.array(a['41']), np.array(a['42']), np.array(a['43']), np.array(a['44']), 
                    np.array(mu['1']), np.array(mu['2']), np.array(mu['3']), np.array(mu['4']), 
                    # np.array(dL_base),
                    
                    d_ln_R1_test, d_ln_R2_test, d_ln_R3_test, d_ln_L_test
                   )

capit_NN = []
for path in results_test:
    capit_NN.append(results_vali[path][0][4][T])

mean_NN = round(np.mean(capit_NN),3)
percentiles = [25, 50, 75]
NN_percentile = np.percentile(capit_NN, percentiles)


print('NeuN mean =', mean_NN)

print('percentile =', round(NN_percentile[0],3), round(NN_percentile[1],3), round(NN_percentile[2],3))

print('Pr(NN loss money)=',total_loss_count_vali/numSim)

print('average return rate monthly = ', (mean_NN/initial_capital)**(1/(T-1))-1)
# print('percentage: number of times the sum of control is too far away from 1 is ', total_counter/numSim)





