import numpy as np
from utils import Trajectory
from TGOSPAMetric import LPTrajMetric_Cluster
from matplotlib import pyplot as plt

#Examples that show how the trajectory GOSPA (T-GOSPA) metric works
#Á. F. García-Fernández, A. S. Rahmathullah and L. Svensson,"A Metric on the Space of Finite Sets of Trajectories for Evaluation of 
#Multi-Target Tracking Algorithms," in IEEE Transactions on Signal Processing, vol. 68, pp. 3917-3928, 2020.

# Authors: Jinhao Gu, Á. F. García-Fernández

delVal=1
DelVal=0.1

def switch(exampleid):
    X=Trajectory()
    Y=Trajectory()
    match exampleid:
        case 1:
            # Example 1
            # T is the total number of time steps
            # stDim is the state dimension
            # nx: number of trajectories in X
            # ny: number of trajectories in Y
            T=5
            stDim=1
            nx=1
            X.xState=np.ones((stDim,T,nx))
            X.tVec=np.array([1])# must be an numpy array
            X.iVec=np.array([T])# must be an numpy array  
            
            ny=1
            Y.xState=X.xState+delVal
            Y.tVec=np.array([1])
            Y.iVec=np.array([T])
        case 2:
            #Example 2
            T=5
            stDim=1
            nx=1
            X.xState=np.ones((stDim,T,nx))
            X.tVec=np.array([1])
            X.iVec=np.array([T])
            
            ny=1
            Y.xState=np.ones((stDim,T,ny))+delVal
            Y.tVec=np.array([1])
            Y.iVec=np.array([T-1])
            
        case 3:    
            T=5
            stDim=1
            nx=1
            X.xState=np.ones((stDim,T,nx))
            X.tVec=np.array([1])
            X.iVec=np.array([T])
            
            ny=2
            Y.xState=np.tile(X.xState+DelVal,(1,1,ny))
            Y.tVec=np.array([1,4])
            Y.iVec=np.array([3,2])
            
        case 4:
            T=5
            stDim=1
            nx=3
            X.xState=np.ones((stDim,T,nx))
            X.tVec=np.array([1,1,4])
            X.iVec=np.array([T,3,2])
            X.xState[:,:,1:3]=X.xState[:,:,1:3]+2*DelVal+delVal
            
            ny=3
            Y.xState=np.tile(X.xState[:,:,0][:, :, np.newaxis]+DelVal,(1,1,ny))
            Y.xState[:,:,2]=Y.xState[:,:,2]+delVal
            
            Y.tVec=np.array([1,4,1])
            Y.iVec=np.array([3,2,T])
            
            
        case 5:
            T=4
            stDim=1
            nx=2
            X.xState=np.ones((stDim,T,nx))
            X.tVec=np.array([1,1])
            X.iVec=np.array([T,T])
            X.xState[:,:,1]=X.xState[:,:,1]+2*DelVal+delVal

            ny=2
            Tswi=3
            Y.xState=np.ones((stDim,T,ny))
            Y.tVec=np.array([1,1])
            Y.iVec=np.array([T,T])
            Y.xState[:,0:Tswi-1,0]=X.xState[:,0:Tswi-1,0]+delVal
            Y.xState[:,Tswi-1:T,0]=X.xState[:,Tswi-1:T,1]-delVal
            Y.xState[:,0:Tswi-1,1]=X.xState[:,0:Tswi-1,1]-delVal
            Y.xState[:,Tswi-1:T,1]=X.xState[:,Tswi-1:T,0]+delVal
        
        
        
        case 6:
            T=4
            stDim=1
            nx=2
            X.xState=np.ones((stDim,T,nx))
            X.tVec=np.array([1,1])
            X.iVec=np.array([T,T])
            X.xState[:,:,1]=X.xState[:,:,1]+2*DelVal+delVal

            ny=3
            Y.xState=np.ones((stDim,T,ny))
            Y.tVec=np.array([1,1,3])
            Y.iVec=np.array([T,2,2])
            Y.xState[:,:,0]=X.xState[:,:,0]+delVal
            Y.xState[:,:,1:3]=np.tile(X.xState[:,:,1][:,:,np.newaxis]-delVal,(1,1,2))
        
        
        case 7:
            
            T=5
            stDim=1
            nx=2
            X.xState=0.5*np.ones((stDim,T,nx))
            X.tVec=np.array([1,1])
            X.iVec=np.array([T,T])
            X.xState[:,:,1]=X.xState[:,:,1]+5*DelVal+delVal
            X.xState[:,2,1]=np.nan

            ny=2
            Y.xState=np.ones((stDim,T,ny))
            Y.tVec=np.array([1,1])
            Y.iVec=np.array([T,T])
            Y.xState[:,:,0]=X.xState[:,:,0]+delVal
            Y.xState[:,:,1]=np.zeros((stDim,T))

            

        case 8:
            #Example with empty trajectory Y
            T=5
            stDim=1
            nx=2
            X.xState=0.5*np.ones((stDim,T,nx))
            X.tVec=np.array([1,1])
            X.iVec=np.array([T,T])
            X.xState[:,:,1]=X.xState[:,:,1]+5*DelVal+delVal
            
            
            
            ny=0
            Y.xState=np.zeros((stDim,T,ny))
            Y.tVec=np.array([])
            Y.iVec=np.array([])
            
            
        
    return X,Y


# Parameters of the metric
c=5
p=1
gamma=1

example_id=7 # [1-8]
X,Y=switch(example_id)
dxy,loc_cost,miss_cost,fa_cost,switch_cost=LPTrajMetric_Cluster(X,Y,c,p,gamma)
print(f'Metric value: {dxy}')
print(f'loc_cost: {np.sum(loc_cost)}, miss_cost: {np.sum(miss_cost)}, fa_cost: {np.sum(fa_cost)}, switch_cost: {np.sum(switch_cost)}')


# We plot the trajectories in X and Y
plt.figure()
nx = len(X.tVec)
for i in range(nx):
    start_time = X.tVec[i]
    end_time = X.iVec[i] + X.tVec[i] - 1
    
    time_range = np.arange(start_time, end_time + 1)
    x_state_values = X.xState[0, start_time-1:end_time, i]
    
    plt.plot(time_range, x_state_values, '-ob')
    
ny = len(Y.tVec)
for i in range(ny):
    start_time = Y.tVec[i]
    end_time = Y.iVec[i] + Y.tVec[i] - 1
    
    time_range = np.arange(start_time, end_time + 1)
    x_state_values = Y.xState[0, start_time-1:end_time, i]
    
    plt.plot(time_range, x_state_values, '-xr')


plt.xlabel('Time step')
plt.ylabel('Target state')
plt.title('Sets of trajectories (X in blue, Y in red)')
plt.grid(True)
plt.show()


T = len(loc_cost)

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



