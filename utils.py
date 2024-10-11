import numpy as np

class Trajectory:
    #This class represents trajectories.
    # Each trajectory is characterised by three parameters
    
    #xState: 'stDim x T x nx' dimensional matrix, where 'stDim' is the
    # single target state dimension, 'T' is the length of the considered time window such that 
    # the metric is computed in the time interval [1,T]
    # The states of trajectory 'ind', 'X.xState[:, :, ind]' has '0' values
    # outisde '[X.tVec[ind], X.tVec[ind]+X.iVec[ind]-1]'. Note that within the
    # window where X.xState is valid, there can be 'holes' in the trajectory, 
    # represented by 'np.nan' values. values.   
   
    # tVec: nx dimensional vector that has start times of the 'nx'
    # trajectories in 'X'. The start time starts in 1.
    # iVec: nx dimensional vector that has the duration of the 'nx' trajectories in 'X'.
  
    def __init__(self,xstate: np.array= None, tvec: np.array=None, ivec: np.array=None):
        
        self.xState=xstate
        self.tVec=tvec
        self.iVec=ivec
        
    
    def getTrajectory(self):
        return self.xState,self.tVec,self.iVec
    
    