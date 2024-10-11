import numpy as np
from utils import Trajectory
from TGOSPAMetric import LPTrajMetric_Cluster
from TW_TGOSPAMetric import TimeWeightedLPTrajMetric_Cluster
from matplotlib import pyplot as plt
from copy import deepcopy

#Examples that show how the time-weighted trajectory GOSPA (T-GOSPA) metric works
#Á. F. García-Fernández, A. S. Rahmathullah and L. Svensson, "A time-weighted metric for sets of trajectories to assess multi-object
#tracking algorithms" in Proceedings of the 24th International Conference on Information Fusion, 2021.

# Author: Á. F. García-Fernández


# Parameters of the metric
c=5


p=1
gamma=10
#We use a rho forgetting factor for the time-weights of the metric
rho=0.995;
# T is the total number of time steps
T = 800

# Time weights for localisation, missed, false targets
# These time weights are normalized to sum to one
time_weights1 = (1 - rho) / (1 - rho**T) * rho**(T - np.arange(1, T + 1))

# Time weights for switching
time_weights2 = time_weights1[1:]



# User can choose the ID of the estimate (from 1 to 4)
ID_est = 3

# Parameters to define sets of trajectories
delVal = 3

# stDim: Target state dimension
# nx: number of trajectories in X
# ny: number of trajectories in Y

stDim = 1
nx = 2
x_far = 100
x_close = 10

#Ground truth trajectory
X=Trajectory()
X.xState = np.zeros((stDim, T, nx))
X.tVec = np.array([1, 1])
X.iVec = np.array([T, T])

X.xState[0, :100, 1] = x_far
X.xState[0, 100:200, 1] = x_close + (x_far - x_close) * (100 - np.arange(100)) / 100
X.xState[0, 200:300, 1] = x_close
X.xState[0, 300:400, 1] = x_close + (x_far - x_close) * np.arange(100) / 100
X.xState[0, 400:500, 1] = x_far
X.xState[0, 500:600, 1] = x_close + (x_far - x_close) * (100 - np.arange(100)) / 100
X.xState[0, 600:700, 1] = x_close
X.xState[0, 700:800, 1] = x_close + (x_far - x_close) * np.arange(100) / 100

# Baseline estimate
ny = 2
Y1=Trajectory()
Y1.xState=np.zeros((stDim, T, ny))
Y1.xState[0, :, 0] = X.xState[0, :, 0] - delVal
Y1.xState[0, :, 1] = X.xState[0, :, 1] + delVal

Y1.tVec = np.array([1, 1])
Y1.iVec = np.array([T, T])

if ID_est == 1:
    # Estimate 1
    Y2=deepcopy(Y1)

elif ID_est == 2:
    # Estimate 2
    Y2=deepcopy(Y1)    
    
    Y2.xState[0, 249:800, 1] = Y1.xState[0, 249:800, 0]  # Track switch
    Y2.xState[0, 249:800, 0] = Y1.xState[0, 249:800, 1]
elif ID_est == 3:
    # Estimate 3
    Y2=deepcopy(Y1)    

    Y2.xState[0, 649:800, 1] = Y1.xState[0, 649:800, 0]  # Track switch
    Y2.xState[0, 649:800, 0] = Y1.xState[0, 649:800, 1]
elif ID_est == 4:
    # Estimate 4
    Y2=deepcopy(Y1)    

    Y2.xState[0, 549:800, 1] = x_far + delVal + (3 * x_far - x_far) * np.arange(1, 252) / 151

# Time weights for localisation, missed, false targets. These time weights are normalised to sum to one
time_weights1 = (1 - rho) / (1 - rho ** T) * rho ** (T - np.arange(1, T + 1))
# Time weights for switching
time_weights2 = time_weights1[1:]

# Plot scenario
plt.figure(1)
plt.clf()

for i in range(nx):
    start_time = X.tVec[i]-1
    end_time = X.iVec[i] + X.tVec[i] - 2
    plt.plot(range(start_time + 1, end_time + 2), X.xState[0, start_time:end_time + 1, i], '-b', linewidth=1.3)

for i in range(ny):
    start_time = Y2.tVec[i] - 1
    end_time = Y2.iVec[i] + Y2.tVec[i] - 2
    plt.plot(range(start_time + 1, end_time + 2), Y2.xState[0, start_time:end_time + 1, i], '-r', linewidth=1.3)

plt.grid(True)
plt.xlabel('Time step')
plt.ylabel('Target state (m)')
plt.title(f'Estimate {ID_est}')
plt.gca().tick_params(labelsize=15)
plt.show()


# TW-T-GOSPA metric computation
dxy_tw,loc_cost_tw,miss_cost_tw,fa_cost_tw,switch_cost_tw=TimeWeightedLPTrajMetric_Cluster(X,Y2,c,p,gamma,time_weights1,time_weights2)

print(f'TW-TGOSPA metric value: {dxy_tw}')
print(f'loc_cost: {np.sum(loc_cost_tw)}, miss_cost: {np.sum(miss_cost_tw)}, fa_cost: {np.sum(fa_cost_tw)}, switch_cost: {np.sum(switch_cost_tw)}')


# T-GOSPA metric computation, normalised by the time window duration
dxy,loc_cost,miss_cost,fa_cost,switch_cost=LPTrajMetric_Cluster(X,Y2,c,p,gamma)
dxy=dxy/T
loc_cost=loc_cost/T
miss_cost=miss_cost/T
fa_cost=fa_cost/T
switch_cost=switch_cost/T
print(f'TGOSPA metric value: {dxy}')
print(f'loc_cost: {np.sum(loc_cost)}, miss_cost: {np.sum(miss_cost)}, fa_cost: {np.sum(fa_cost)}, switch_cost: {np.sum(switch_cost)}')


# We plot the metric decompositions
# Overall TW-T-GOSPA metric decomposed against time steps
dxy_tw_time = np.power(loc_cost_tw + miss_cost_tw + fa_cost_tw + np.concatenate((np.zeros([1,1]), switch_cost_tw)), 1/p)

# Define the time range (T)

plt.figure()
plt.plot(np.arange(1, T+1), dxy_tw_time, 'black',label='Total')
plt.plot(np.arange(1, T+1), np.power(loc_cost_tw, 1/p), '--r',label='Localisation')
plt.plot(np.arange(1, T+1), np.power(miss_cost_tw, 1/p), '-.g',label='Missed')
plt.plot(np.arange(1, T+1), np.power(fa_cost_tw, 1/p), '-*b',label='False')
plt.plot(np.arange(1, T+1), np.power(np.append([0], switch_cost_tw), 1/p), '-+m', label='Switch')

# Enable grid
plt.grid(True)
plt.legend(loc="upper left")
plt.ylabel('Time weighted LP T-GOSPA metric decomposition')
plt.xlabel('Time step')
plt.plot()




# Overall T-GOSPA metric decomposed against time steps
dxy_time = np.power(loc_cost + miss_cost + fa_cost + np.concatenate((np.zeros([1,1]), switch_cost)), 1/p)

# Define the time range (T)

plt.figure()
plt.plot(np.arange(1, T+1), dxy_time, 'black',label='Total')
plt.plot(np.arange(1, T+1), np.power(loc_cost, 1/p), '--r',label='Localisation')
plt.plot(np.arange(1, T+1), np.power(miss_cost, 1/p), '-.g',label='Missed')
plt.plot(np.arange(1, T+1), np.power(fa_cost, 1/p), '-*b',label='False')
plt.plot(np.arange(1, T+1), np.power(np.append([0], switch_cost), 1/p), '-+m', label='Switch')

# Enable grid
plt.grid(True)
plt.legend(loc="upper left")
plt.ylabel('LP T-GOSPA metric decomposition')
plt.xlabel('Time step')
plt.plot()



