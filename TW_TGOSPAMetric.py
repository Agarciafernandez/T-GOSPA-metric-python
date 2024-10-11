#This code is a python implementation of the time-weighted trajectory GOSPA (TW-T-GOSPA) metric between sets of trajectories.
# In particular, it corresponds to the linear programming (LP) implementation in

#Á. F. García-Fernández, A. S. Rahmathullah and L. Svensson, "A time-weighted metric for sets of trajectories to assess multi-object
#tracking algorithms" in Proceedings of the 24th International Conference on Information Fusion, 2021.

# The code is based on a translation of the MATLAB implementation of the LP TW-T-GOSPA metric 
# LPTrajMetric_cluster.m available at 
# https://github.com/Agarciafernandez/MTT/tree/master/Trajectory%20metric
# The authors of this Python implementation are Jinhao Gu and Á. F. García-Fernández

import numpy as np 
from utils import Trajectory
import scipy.sparse as sps
from scipy.optimize import linprog

def computeLocCostPerTime(x,y,c,p):
    if np.all(np.invert(np.isnan(x))) and np.all(np.invert(np.isnan(y))):
        #neither x nor y has hole
        d=min(np.linalg.norm(x-y,p)**p,c**p)
    
    elif np.any(np.logical_and(np.isnan(x),np.invert(np.isnan(y)))) or np.any(np.logical_and(np.isnan(y),np.invert(np.isnan(x)))): 
        #exactly one of x and y has hole 
        d=c**p/2
    else:
        d=0
        
    return d

def locCostComp_V2(X,Y,c,p):
    '''
    computing the localisation cost at each time 't' for every (i,j)
    '''
    
    tmpCost=c**p/2# cost for being unassigned
    T=X.xState.shape[1]
    nx=X.xState.shape[2]
    ny=Y.xState.shape[2]
    locCostMat=np.zeros((nx+1,ny+1,T))
    for t in range(T):
        for xind in range(nx+1):
            if xind <=nx-1: # xind not dummy
                if t+1 >= X.tVec[xind] and t+1 <= (X.tVec[xind]+X.iVec[xind]-1) and not np.isnan(X.xState[0,t,xind]):
                # if X_xind exists at time t
                    for yind in range(ny+1):
                            if yind <=ny-1 and t+1>=Y.tVec[yind] and t+1<=Y.iVec[yind]+Y.tVec[yind]-1:
                                # if Y_yind exists at time t
                                locCostMat[xind,yind,t]=computeLocCostPerTime(X.xState[:,t,xind],Y.xState[:,t,yind],c,p)
                                
                            else:
                                locCostMat[xind,yind,t]=tmpCost
                else:#yind does not exist or yind is dummy
                    for yind in range(ny):
                        if t+1>=Y.tVec[yind] and t+1<=Y.iVec[yind]+Y.tVec[yind]-1 and not np.isnan(Y.xState[0,t,yind]):
                            # if Y_yind exists at time t    
                            locCostMat[xind,yind,t]=tmpCost
                        
                        
            else:    #  xind is dummy
                for yind in range(ny):
                    if t+1>=Y.tVec[yind] and t+1<=Y.iVec[yind]+Y.tVec[yind]-1 and not np.isnan(Y.xState[0,t,yind]):
                    # if Y_yind exists at time t    
                        locCostMat[xind,yind,t]=tmpCost
    return locCostMat
                     

def computeLocFalseMissedSwitchCosts(wMat,LocCostMat,X,Y,c,p,gamma):
    '''
    Computing the localisation cost, switching cost and cost for missed and false targets
    '''
    tmp_cost=c**p/2# cost for being unassigned
    T=X.xState.shape[1]
    nx=X.xState.shape[2]
    ny=Y.xState.shape[2]

    if (nx>1 and ny>1):
        switch_cost=0.5*gamma**p *np.sum(np.sum(abs(np.diff(wMat[:nx,:ny,:],1,2)),axis=1),axis=0)
    elif nx==0 or ny==0:
        switch_cost=np.zeros((T-1,1))
    elif nx==1 and ny==1:
        switch_cost = 0.5*gamma**p * abs(np.diff(wMat[:nx,:ny,:],1,2))
    else:
        switch_cost=0.5*gamma**p * np.sum(abs(np.squeeze(np.diff(wMat[:nx,:ny,:],1,2))),axis=0)


    loc_mask=np.zeros(wMat.shape)
    miss_mask=np.zeros(wMat.shape)
    fa_mask=np.zeros(wMat.shape)
    fa_miss_mask=np.zeros(wMat.shape)#accounts for false and missed target costs that arise for a localisation cost of c^p

    for t in range(T):
        for xind in range(nx+1): #xind not dummy
            if xind<nx:
                if (t+1>=X.tVec[xind] and t+1<=X.tVec[xind]+X.iVec[xind]-1) and not np.isnan(X.xState[0,t,xind]):
                    # if X_xind exists at time t
                    for yind in range(ny+1):
                            if yind <ny and (t+1>=Y.tVec[yind] and t+1<=Y.tVec[yind]+Y.iVec[yind]-1) and not np.isnan(Y.xState[0,t,yind]):
                                # if Y_yind exists at time t
                                ## add to localisation cost at the time based on weight (unless the weight is c^p) 
                                if LocCostMat[xind,yind,t]<2*tmp_cost:
                                    loc_mask[xind,yind,t]=1
                                else:
                                    fa_miss_mask[xind,yind,t]=1
                            else:#  yind does not exist or yind is dummy
                                miss_mask[xind,yind,t]=1
                else:# if X_xind does not exist at time t
                    for yind in range(ny):
                        if t+1>=Y.tVec[yind] and t+1<=Y.tVec[yind]+Y.iVec[yind]-1 and not np.isnan(Y.xState[0,t,yind]):
                            # if Y_yind exists at time t
                            fa_mask[xind,yind,t]=1
            else: # xind is dummy
                for yind in range(ny):
                    if t+1>=Y.tVec[yind] and t+1<=Y.tVec[yind]+Y.iVec[yind]-1 and not np.isnan(Y.xState[0,t,yind]):
                        # if Y_yind exists at time t
                        fa_mask[xind,yind,t]=1
                        
    loc_cost=np.sum(np.sum(LocCostMat*wMat*loc_mask,axis=1),axis=0)
    miss_cost=tmp_cost*np.sum(np.sum(miss_mask*wMat,axis=1),axis=0)+tmp_cost*np.sum(np.sum(fa_miss_mask*wMat,axis=1),axis=0)
    fa_cost=tmp_cost*np.sum(np.sum(fa_mask*wMat,axis=1),axis=0)+tmp_cost*np.sum(np.sum(fa_miss_mask*wMat,axis=1),axis=0)
    
    return loc_cost,miss_cost,fa_cost,switch_cost


                     
def TimeWeightedLP_metric_cluster(X,Y,DAB,nx,ny,nxny,nxny2,T,c,p,gamma,time_weights1,time_weights2):
    ''' 
    #### Variables to be calculated in LP ####
    x = [W_1,1(1) W_2,1(1) .. W_nx+1,1(1), .. W_1,ny+1(1) W_2,ny+1(1) ...
    W_nx+1,ny+1(1), W_1,1(T) W_2,1(T) .. W_nx+1,1(T), .. W_1,ny+1(T)
    W_2,ny+1(T) ... W_nx+1,ny+1(T) e(1) .. e(T-1) h_11(1) .. h_nx,ny(1) ...
    h_1,1(T-1) ... h_nx,ny(T-1)]'
    '''
    #### Length of the variable components in x 
    WtLen=nxny2*T
    etLen=T-1
    htLen=nxny*(T-1)
    nParam=WtLen+etLen+htLen #total number of variables
    
    #### Position of the variable components in x
    WtPos=np.arange(0,WtLen)
    etPos=np.arange(WtLen,WtLen+etLen)
    htPos=np.arange(WtLen+etLen,nParam)
    
    ##########################################
    
    #### objective function f ####
    
    #Time weighted DAB
    time_weights1_tensor=np.tile(np.squeeze(time_weights1), (DAB.shape[0], DAB.shape[1],1))
    
    DAB_TW = DAB * time_weights1_tensor
    
    
    f=np.zeros((nParam,1))
    f[WtPos]=np.reshape(DAB_TW,(WtLen,1),order='F')# for vec(W(1)) to vec(W(T))
    f[etPos]=0.5*gamma**p * time_weights2
    
    
    ##### equality constraints
    #Consstraint 1
    
    index_x=np.tile(np.arange(0,T*ny),(nx+1,1))
    index_x=index_x.flatten(order='F')
    index_y=np.zeros((T*ny*(nx+1),1))
    index_rep=np.arange(ny*(nx+1))
    # index_rep
    for i in range(T):
        index_y[index_rep+i*len(index_rep)]=(ny+1)*(nx+1)*i+index_rep.reshape(-1,1)
    index_y=index_y.flatten(order='F')

    Aeq1=sps.coo_matrix((np.ones(len(index_x)),(index_x,index_y)),shape=(ny*T,nParam))
    beq1=np.ones((ny*T,1))
    
    
    
    #Constraint 2
    index_x=np.tile(np.reshape(np.arange(T*nx),(nx,T),order='F'),(ny+1,1))
    index_x=index_x.flatten(order='F')
    index_y=np.tile(np.arange(1,(nx+1)/nx*len(index_x),step=nx+1),(nx,1))-1
    index_y2=np.tile(np.arange(nx),(np.size(index_y,1),1)).T
    index_y=index_y+index_y2
    index_y=index_y.flatten(order='F')
    Aeq2=sps.coo_matrix((np.ones(len(index_x)),(index_x,index_y)),shape=(nx*T,nParam))


    beq2=np.ones((nx*T,1))
    Aeq=sps.vstack([Aeq1,Aeq2])
    beq=np.vstack([beq1,beq2])
    ##################################
    
    #### uppper and lower bounds constraints ####
    lb=np.zeros(nParam)
    ub=np.inf*np.ones(nParam)
    bounds=[(lb[i],ub[i]) for i in range(nParam)]
    
    #################################
    
    
    #### Inequality constraints ####
    
    # Inequality constraint 1
    index_minus_x=np.arange(T-1)
    index_minus_y=WtLen+index_minus_x
    value_minus=-1*np.ones((T-1))
    index_one_x=np.tile(np.arange(T-1),(nxny,1))
    index_one_x=index_one_x.flatten(order='F')
    index_one_y=np.arange(WtLen+etLen,WtLen+etLen+(T-2)*nxny+nxny)
    value_one=np.ones((len(index_one_y)))
    index_x_=np.hstack((index_minus_x,index_one_x))
    index_y_=np.hstack((index_minus_y,index_one_y))
    value_=np.hstack((value_minus,value_one))
    A1=sps.coo_matrix((value_,(index_x_,index_y_)),shape=(T-1,nParam))
    
    # Inequality constraint 2
    index_m1_x=np.arange(nxny*(T-1))
    index_m1_y=htPos
    index_1_x=index_m1_x
    index_y=np.tile(np.arange(1,(nx+1)*(ny+1)*(T-1)+1,step=nx+1),(nx,1))-1
    index_y=np.delete(index_y,np.arange(ny,index_y.shape[1],step=ny+1),axis=1)
    index_1_y=index_y+np.tile(np.arange(nx),(index_y.shape[1],1)).T
    index_1_y=index_1_y.flatten(order='F')
    index_2_x=index_1_x
    index_2_y=index_1_y+(nx+1)*(ny+1)
    index_2x_=np.hstack((index_1_x,index_m1_x,index_2_x))
    index_2y_=np.hstack((index_1_y,index_m1_y,index_2_y))
    value_2_=np.hstack((np.ones(len(index_1_y)),-np.ones(len(index_m1_y)),-np.ones(len(index_2_y))))
    A3=sps.coo_matrix((value_2_,(index_2x_,index_2y_)),shape=(nxny*(T-1),nParam))

    # Inequality constraint 3
    value_3_=np.hstack((-np.ones(len(index_1_y)),-np.ones(len(index_m1_y)),np.ones(len(index_2_y))))
    A4=sps.coo_matrix((value_3_,(index_2x_,index_2y_)),shape=(nxny*(T-1),nParam))

    # All Inequality constraints
    A=sps.vstack([A1,A3,A4])
    b=np.zeros(((T-1)+nxny*(T-1)+nxny*(T-1),1))
    
    #optimisation
    res=linprog(f,A_ub=A,b_ub=b,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs')
    dxy=res.fun
    
    #Metric and the assignment values to be returned
    wMat=np.reshape(res.x[0:nxny2*T],(nx+1,ny+1,T),order='F')
    loc_cost,miss_cost,fa_cost,switch_cost=computeLocFalseMissedSwitchCosts(wMat,DAB,X,Y,c,p,gamma)
    
    #We multiply for the time-weights the different costs
    time_weights1_s=np.squeeze(time_weights1)
    time_weights2_s=np.squeeze(time_weights2)

    
    loc_cost=loc_cost*time_weights1_s
    miss_cost=miss_cost*time_weights1_s
    fa_cost=fa_cost*time_weights1_s
    switch_cost=switch_cost*time_weights2_s
    
    return dxy,loc_cost,miss_cost,fa_cost,switch_cost



def TimeWeightedLPTrajMetric_Cluster(X,Y,c,p,gamma,time_weights1,time_weights2):
    '''
    This function computes the time weighted trajectory generalised optimal subpattern assignment metric (T-GOSPA) between sets of trajectories.
    In particular, it uses the linear programming (LP) implementation in
    in Á. F. García-Fernández, A. S. Rahmathullah and L. Svensson, "A time-weighted metric for sets of trajectories to assess multi-object
    tracking algorithms" in Proceedings of the 24th International Conference on Information Fusion, 2021.
    ----------------------------------------
    Input:
     X, Y: sets of trajctories which are structs as follows:
       X.tVec: nx dimensional vector that has start times of the 'nx'
           trajectories in 'X'.
       X.iVec: nx dimensional vector that has the duration of the 'nx'
           trajectories in 'X'.
       X.xState: 'stDim x T x nx' dimensional matrix, where 'stDim' is the
           state dimension, 'T' is the length of the considered time window such that 
           the metric is computed in the time interval [1,T]. The
           states of trajectory 'ind', 'X.xState[:, :, ind]' has '0' values
           outisde '[X.tVec[ind], X.tVec[ind]+X.iVec[ind]-1]'. Note that within the
           window where X.xState is valid, there can be 'holes' in the trajectory, 
           represented by 'np.nan' values.
     c: >0, cut-off parameter
     p: >= 1, exponent parameter
     gamma: >0, track switch penalty
     
     time_weights1 is the time weight vector of length T for the
     localisation/missed/false target costs
     time_weights2 is the time weight vector of length T-1 for the
     switching costs
    ----------------------------------------
    Output:
    dxy: Metric value
    loc_cost: localisation cost (to the p-th power) for properly detected targets over time of dimension 'T x 1'
    miss_cost: cost (to the p-th power) for missed targets over time, dimension 'Tx1'
    fa_cost: cost (to the p-th power) for false targets over time, dimension 'Tx1'
    switch_cost: cost (to the p-th power) for switches over time, dimension '(T-1)x1'
    ----------------------------------------
    '''
    
    time_weights1=np.expand_dims(time_weights1, axis=1)
    time_weights2=np.expand_dims(time_weights2, axis=1)

    #Input parameters
    nx=X.xState.shape[2]
    ny=Y.xState.shape[2]
    T=X.xState.shape[1]
    
    ################
    if nx ==0 and ny==0:
        dxy=0
        loc_cost=np.zeros([T,1])
        miss_cost=np.zeros([T,1])
        fa_cost=np.zeros([T,1])
        switch_cost=np.zeros([T-1,1])
        return dxy,loc_cost,miss_cost,fa_cost,switch_cost
    
    if len(time_weights1) != T or len(time_weights2) != T - 1:
        raise ValueError('Time weights in time weighted LP trajectory metric do not have the correct dimensions')
    
    #Localisation cost computation
    DAB=locCostComp_V2(X,Y,c,p)
    
    #Clustering
    G=np.zeros((nx,ny))
    T_min_a=-np.ones((nx,ny))#Matrices with minimum and maximum times in which trajectories i and j can be associated
    T_max_a=-np.ones((nx,ny))
    for i in range(nx):
        t_i_x=X.tVec[i]
        t_f_x=X.tVec[i]+X.iVec[i]-1
        for j in range(ny):
            t_i_y=Y.tVec[j]
            t_f_y=Y.tVec[j]+Y.iVec[j]-1
            t_i_max=max(t_i_x,t_i_y)
            t_f_min=min(t_f_x,t_f_y)
            #option 1
            G_ij=np.squeeze(DAB[i,j,t_i_max-1:t_f_min])>=c**p
            isnan_ij=np.transpose(np.logical_or(np.isnan(X.xState[0,t_i_max-1:t_f_min,i]), \
                                                np.isnan(Y.xState[0,t_i_max-1:t_f_min,j])))#We need to check when any of the trajectories isnan in this interval
            sum_isnan_ij=np.sum(isnan_ij)
            G[i,j]=np.sum(G_ij)==max(t_f_min-t_i_max+1-sum_isnan_ij,0)
            t_min_ij=np.where(np.logical_and(G_ij==0,isnan_ij==0))[0]+1#They can be associated if this value is zero

            if len(t_min_ij):
                T_min_a[i,j]=t_min_ij[0]+t_i_max-1
                T_max_a[i,j]=t_min_ij[-1]+t_i_max-1
            else:
                T_min_a[i,j]=T+1#We put infeasible values
                T_max_a[i,j]=-1
                
    #Graph connectivity
    G_con=np.logical_not(G[:,:])
    Adj_matrix=np.vstack((np.hstack((np.eye(nx),G_con)),np.hstack((G_con.T,np.eye(ny)))))
    #Clustering
    r=np.flip(sps.csgraph.reverse_cuthill_mckee(sps.csr_matrix(Adj_matrix)))

    Clusters=[[r[0]]]
    max_length_c=1

    for i in range(1,len(r)):
        if any(Adj_matrix[Clusters[-1],r[i]]):
            Clusters[-1].append(r[i])
        else:
            Clusters.append([r[i]])
        if len(Clusters[-1])>max_length_c:
            max_length_c=len(Clusters[-1])



    loc_cost=np.zeros([T,1])
    miss_cost=np.zeros([T,1])
    fa_cost=np.zeros([T,1])
    switch_cost=np.zeros([T-1,1])
    dxy=0
    for i in range(len(Clusters)):
        Cluster_i=np.array(Clusters[i])
        if (len(Cluster_i)==1):
            #only one trajectory in the cluster
            i_x=np.where(Cluster_i<nx)[0]
            if len(i_x)==0:
                # Then it is a trajectory in Y
                i_y=np.where(Cluster_i>=nx)[0]
                list_y=Cluster_i[i_y]-nx
                t_axis=np.arange(Y.tVec[list_y],Y.tVec[list_y]+Y.iVec[list_y])-1
                isrealY_i=~np.isnan(Y.xState[0:1,t_axis,list_y])
                #dxy_i=c**p/2*np.sum(np.squeeze(isrealY_i))
                #fa_cost[t_axis]=fa_cost[t_axis]+c**p/2*isrealY_i.T
                
                dxy_i = (c**p / 2) * np.sum(time_weights1[isrealY_i])
                fa_cost[t_axis] = fa_cost[t_axis]+ (c**p / 2) * time_weights1[isrealY_i]
                
                
            else:
                list_x=Cluster_i[i_x]
                t_axis=np.arange(X.tVec[list_x],X.tVec[list_x]+X.iVec[list_x])-1
                isrealX_i=~np.isnan(X.xState[0:1,t_axis,list_x])
                #dxy_i=c**p/2*np.sum(np.squeeze(isrealX_i))
                #miss_cost[t_axis]=miss_cost[t_axis]+c**p/2*isrealX_i.T
                
                dxy_i = (c**p / 2) * np.sum(time_weights1[isrealX_i])
                miss_cost[t_axis] = miss_cost[t_axis] + (c**p / 2) * time_weights1[isrealX_i]
            
        elif (len(Cluster_i)==2):
            i_x=np.where(Cluster_i<nx)[0]
            list_x=Cluster_i[i_x]
            i_y=np.where(Cluster_i>=nx)[0]
            list_y=Cluster_i[i_y]-nx
            
            
            DAB_i = DAB[list_x, list_y, :]
            dxy_i = np.sum(DAB_i.T * time_weights1)
            
            
            t_axis=np.arange(X.tVec[list_x],X.tVec[list_x]+X.iVec[list_x])-1
            isnan_X=np.isnan(X.xState[0,:,list_x])
            no_exist_X=np.ones((T,1))
            no_exist_X[t_axis]=0
            
            
            t_axis=np.arange(Y.tVec[list_y],Y.tVec[list_y]+Y.iVec[list_y])-1
            isnan_Y=np.isnan(Y.xState[0,:,list_y])
            no_exist_Y=np.ones((T,1))
            no_exist_Y[t_axis]=0
            
            DAB_i=DAB[list_x,list_y,:]
            
            #Errors equal to c^p correspond to false and missed target costs
            index1=np.squeeze(DAB_i==c**p)
            #fa_cost[index1]=fa_cost[index1]+c**p/2 
            #miss_cost[index1]=miss_cost[index1]+c**p/2

            fa_cost[index1]=fa_cost[index1]+c**p/2 *time_weights1[index1];
            miss_cost[index1]=miss_cost[index1]+c**p/2*time_weights1[index1];

            #Missed targets
            index2=np.squeeze(np.logical_and(DAB_i==c**p/2,np.logical_or(isnan_X,np.transpose(no_exist_X))))
            #fa_cost[index2]=fa_cost[index2]+c**p/2
            
            fa_cost[index2]=fa_cost[index2]+c**p/2*time_weights1[index2]

            
            #False targets
            index3=np.squeeze(np.logical_and(DAB_i==c**p/2,np.logical_or(isnan_Y,np.transpose(no_exist_Y))))
            #miss_cost[index3]=miss_cost[index3]+c**p/2
            miss_cost[index3]=miss_cost[index3]+c**p/2*time_weights1[index3]
            
            # Localisation cost
            index4=np.squeeze(np.logical_and(DAB_i>0,np.logical_not(np.logical_or(index1,np.logical_or(index2,index3)))))
            #loc_cost[index4]=loc_cost[index4]+DAB_i[:,index4].T
            
            loc_cost[index4]=loc_cost[index4]+DAB_i[:,index4].T*time_weights1[index4];

        
        else:
            i_x=np.where(Cluster_i<nx)[0]
             
            list_x=Cluster_i[i_x]
            i_y=np.where(Cluster_i>=nx)[0]
            
            list_y=Cluster_i[i_y]-nx
            
            X_i=Trajectory()
            X_i.tVec=X.tVec[list_x]
            X_i.iVec=X.iVec[list_x]
            
            Y_i=Trajectory()
            Y_i.tVec=Y.tVec[list_y]
            Y_i.iVec=Y.iVec[list_y]
            
            #Calculate minimum time and maximum times when we need to consider the assignments
            t_min=int(np.min(T_min_a[list_x][:,list_y]))
            t_max=int(np.max(T_max_a[list_x][:,list_y]))
            
            tf_X=X_i.tVec+X_i.iVec-1
            tf_Y=Y_i.tVec+Y_i.iVec-1
            
            #We sum the costs outside the cosidered window
            miss_cost_i=np.zeros(np.size(miss_cost))
            fa_cost_i=np.zeros(np.size(fa_cost))
            isrealX_i=~np.isnan(X.xState[0:1,:,list_x])
            isrealY_i=~np.isnan(Y.xState[0:1,:,list_y])
            
            for j in range(len(X_i.tVec)):
                #We add a -1 to account for difference in Python indexing w.r.t. Matlab
                t_axis=[x-1 for x in range(int(X_i.tVec[j]),int(t_min))]+[x-1 for x in np.arange(int(t_max+1),int(tf_X[j])+1)]
                
                #miss_cost_i[t_axis]=miss_cost_i[t_axis]+c**p/2*np.squeeze(isrealX_i[0:1,t_axis,j])
            
                miss_cost_i[t_axis]=miss_cost_i[t_axis]+c**p/2*np.squeeze(isrealX_i[0:1,t_axis,j])*time_weights1[np.squeeze(isrealX_i[0:1,t_axis,j])]
            
            for j in range(len(Y_i.tVec)):
                t_axis=[x-1 for x in range(int(Y_i.tVec[j]),int(t_min))]+[x-1 for x in np.arange(int(t_max+1),int(tf_Y[j])+1)]
                #fa_cost_i[t_axis]=fa_cost_i[t_axis]+c**p/2*np.squeeze(isrealY_i[0:1,t_axis,j])
            
                fa_cost_i[t_axis]=fa_cost_i[t_axis]+c**p/2*np.squeeze(isrealY_i[0:1,t_axis,j])*time_weights1[np.squeeze(isrealY_i[0:1,t_axis,j])]

            X_i.xState=X.xState[:,t_min-1:t_max,list_x]
            Y_i.xState=Y.xState[:,t_min-1:t_max,list_y]
            
            ti_X=X_i.tVec
            ti_Y=Y_i.tVec
            
            X_i.tVec=np.maximum(ti_X-t_min+1,1)
            Y_i.tVec=np.maximum(ti_Y-t_min+1,1)
            
            X_i.iVec=X_i.iVec-np.maximum(t_min-ti_X,0)-np.maximum(tf_X-t_max-1,0)
            Y_i.iVec=Y_i.iVec-np.maximum(t_min-ti_Y,0)-np.maximum(tf_Y-t_max-1,0)
            T_i=t_max-t_min+1
            
            # DAB_i=DAB
            DAB_i=DAB[np.ix_(np.append(list_x,nx),np.append(list_y,ny),np.arange(t_min-1,t_max))]
            nx_i=len(list_x)
            ny_i=len(list_y)
            nxny_i=nx_i*ny_i
            nxny2_i=(nx_i+1)*(ny_i+1)
            dxy_i,loc_cost_i2,miss_cost_i2,fa_cost_i2,switch_cost_i2=TimeWeightedLP_metric_cluster(X_i,Y_i,DAB_i,nx_i,ny_i,nxny_i,nxny2_i,T_i,c,p,gamma,time_weights1[t_min-1:t_max],time_weights2[t_min-1:t_max-1])
        
            #We add the false and missed targe costs of the tails
            dxy_i=dxy_i+np.sum(miss_cost_i)+np.sum(fa_cost_i);

            
        
            miss_cost_i=np.reshape(miss_cost_i,(T,1))
            fa_cost_i = np.reshape(fa_cost_i,(T,1))
            miss_cost_i[t_min-1:t_max]=miss_cost_i[t_min-1:t_max]+np.reshape(miss_cost_i2,(T_i,1))
            fa_cost_i[t_min-1:t_max]=fa_cost_i[t_min-1:t_max]+np.reshape(fa_cost_i2,(T_i,1))
            loc_cost[t_min-1:t_max]=loc_cost[t_min-1:t_max]+np.reshape(loc_cost_i2,(T_i,1))
            switch_cost[t_min-1:t_max-1]=switch_cost[t_min-1:t_max-1]+np.reshape(switch_cost_i2,(T_i-1,1))
            miss_cost=miss_cost+miss_cost_i
            fa_cost=fa_cost+fa_cost_i
            
        dxy=dxy+dxy_i

    dxy=dxy**(1/p)

    return dxy,loc_cost,miss_cost,fa_cost,switch_cost

            