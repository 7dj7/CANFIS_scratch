import numpy as np
import numpy.matlib
import pandas as pd
import time
from math import *
import sys

class Architecture:
    def __init__(self, config, mparams, kparams, nodes, ni, no, mf, nc, last_decrease_ss = 1, last_increase_ss = 1):
        self.config = config
        self.mparams = mparams
        self.kparams = kparams
        self.nodes = nodes
        self.ni = ni
        self.no = no
        self.mf = mf
        self.nc = nc
        self.last_decrease_ss = last_decrease_ss
        self.last_increase_ss = last_increase_ss
        self.S = []
        self.P = []

def calculate_output1(mynet):
    mparams = mynet.mparams
    for k in range(mynet.no):
        for i in range(mynet.ni):
            for j in range(mynet.mf):
                ind = mynet.ni + i*mynet.mf + j
                x = mynet.nodes[i,k]
                sigma = mparams[i*mynet.mf+j,0,k]
                if np.isnan(sigma):
                    print('in calculate output1 prob!')
                    time.sleep(0.2)
                c = mparams[i*mynet.mf+j,0,k]
                if sigma == 0:
                    tmp1 = 0
                else: 
                    tmp1 = (x - c)/sigma
                if sigma == 0:
                    tmp2 = 0
                else:
                    tmp2 = tmp1*tmp1
                if sigma == 0:
                    tmp = 0
                else:
                    tmp = np.exp(-0.5 *tmp2)
                mynet.nodes[ind,k] = tmp                    # gaussmf
    return mynet

def calculate_output2(mynet):
    st = mynet.ni + mynet.ni*mynet.mf
    for i in range(st,st+mynet.nc):
        I = np.where(mynet.config[:,i] == 1.0)
        for k in range(mynet.no):
            tmp = np.cumprod(mynet.nodes[I,k]) 
            mynet.nodes[i,k] = tmp[-1]
    return mynet

def calculate_output3(mynet):
    st = mynet.ni + mynet.ni*mynet.mf + mynet.nc 
    for i in range(st,st+mynet.nc):
        I = np.where(mynet.config[:,i] == 1)
        for k in range(mynet.no):
            denom = sum(sum(mynet.nodes[I,k]))
            mynet.nodes[i,k] = mynet.nodes[i-mynet.nc,k]/denom
    return mynet

def calculate_output4(mynet):
    st = mynet.ni + mynet.ni*mynet.mf + 2*mynet.nc 
    kparam = mynet.kparams
    for k in range(mynet.no):
        inp = mynet.nodes[0:mynet.ni, k]
        for i in range(mynet.nc):
            wn = mynet.nodes[i+st-mynet.nc, k]
            mynet.nodes[i+st,k] = wn*(sum(np.multiply(kparam[i,0:-1,k], inp)) + kparam[i,-1,k])
    return mynet

def calculate_output5(mynet):
    for k in range(mynet.no):
        mynet.nodes[-1,k] = sum(mynet.nodes[-mynet.nc-1:-1, k])
    return mynet

def get_kalman_data(mynet, outputs):   
    kalman_data = np.zeros(((mynet.ni+1)*mynet.nc+1,mynet.no))
    st = mynet.ni + mynet.ni*mynet.mf + mynet.nc
    for ii in range(mynet.no):
        j = 0
        for i in range(st,st+mynet.nc):
            for k in range(mynet.ni):
                kalman_data[j,ii] = mynet.nodes[i,ii]*mynet.nodes[k,ii]
                j = j + 1
            kalman_data[j,ii] = mynet.nodes[i,ii]
            j = j + 1
        kalman_data[j,ii] = outputs[ii]
    return kalman_data

def mykalman(mynet, kalman_data, j):   
    for ii in range(mynet.no):
        k_p_n = (mynet.ni + 1) * mynet.nc
        alpha = 1000000
        if j==0:
            mynet.P = np.zeros((k_p_n,1))
            mynet.S = alpha*np.eye(k_p_n)
        x = kalman_data[0:-1,ii]  
        y = kalman_data[-1,ii]  
        x.shape = (21,1)
        tmp1 = np.transpose(np.matmul(np.transpose(x),mynet.S))
        denom = 1 + sum(sum(np.multiply(tmp1, x)))
        tmp2 = tmp1
        tmp1 = np.matmul(mynet.S,x)

        # tmp2 = np.transpose((np.transpose(x)*mynet.S))
        tmp_m = np.matmul(tmp1,np.transpose(tmp2))
        tmp_m = (-1/denom)*tmp_m
        mynet.S = mynet.S + tmp_m

        diff = y - sum(sum(np.multiply(x, mynet.P)))
        tmp1 = diff*(np.matmul(mynet.S,x))
        mynet.P = mynet.P + tmp1
        mynet.kparams[:,:,ii] = np.transpose(mynet.P.reshape(mynet.ni+1, mynet.nc))
    return mynet

def clear_de_dp(mynet):
    mynet.mparam_de_do = np.zeros((mynet.ni*mynet.mf,2,mynet.no))
    mynet.kparam_de_do = np.zeros((mynet.nc,mynet.ni+1, mynet.no))
    return mynet
# equivalent function of matlab find returning indices of array with matching condition
def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def do4_do3(mynet,i, j, k):
    kparam = mynet.kparams
    inp = np.transpose(mynet.nodes[0:mynet.ni, k])
    jj = j - mynet.ni - mynet.ni*mynet.mf - 2*mynet.nc
    tmp = sum(np.multiply(kparam[jj, 0:-1, k], inp)) + kparam[jj,-1,k]
    if np.isnan(tmp):
        print('problem in do4_do3 derivative')
    return tmp

def do3_do2(mynet,i, j, k):
    II = indices(mynet.config[:,j], lambda x: x == 1)
    I = indices(II, lambda x : x < j)
    m = [II[i] for i in I]
    n = [mynet.nodes[j,k] for j in m]
    total = sum(n)
    if j-i == mynet.nc:
        tmp = (total-mynet.nodes[i,k])/(total*total)
    else:
        tmp = -mynet.nodes[j - mynet.nc, k]/(total*total)
    if np.isnan(tmp):
        print('problem in do3_do2 derivative')
    return tmp

def derivative_o_o(mynet,i, j, k):
    if i >= mynet.ni + mynet.ni*mynet.mf + 2*mynet.nc:
        tmp = 1
    elif i >= mynet.ni + mynet.ni*mynet.mf + mynet.nc:
        tmp = do4_do3(mynet, i, j, k)
    elif i >= mynet.ni + mynet.ni*mynet.mf:
        tmp = do3_do2(mynet, i, j, k)
    elif i >= mynet.ni:
        if mynet.nodes[i,k] == 0:
            tmp = 0
        else:
            tmp = mynet.nodes[j,k]/mynet.nodes[i,k]
    return tmp


def calculate_de_do(mynet, de_dout):
    mynet.de_do = np.zeros((np.size(mynet.nodes, 0), mynet.no))
    for k in range(mynet.no):   
        mynet.de_do[-1,k] = de_dout[k]
        for i in range(len(mynet.nodes[:,k])-2, mynet.ni + 1,-1):
            de_do = 0
            II = indices(mynet.config[i,:], lambda x: x == 1)
            I = indices(II,lambda x: x > i)
            for j in range(len(I)):
                jj = II[I[j]]
                tmp1 = mynet.de_do[jj, k]
                tmp2 = derivative_o_o(mynet,i, jj, k)
                de_do = de_do + tmp1*tmp2
            mynet.de_do[i,k] = de_do
    return mynet

def dmf_dp(mynet,i, j, k):
    I = indices(mynet.config[:,i], lambda x: x == 1)
    x=mynet.nodes[I, k]
    sigma = mynet.mparams[i-mynet.ni, 0, k]
    c = mynet.mparams[i-mynet.ni, 1, k]
    if np.isnan(sigma):
        print('in dmf_dp prob!')
        time.sleep(2)
    ### gaussmf  
    if sigma == 0:
        tmp1 = 0
    else:
        tmp1 = (x[0] - c)/sigma
    if sigma == 0:
        tmp2 = 0
    else:
        tmp2 = exp(-0.5*tmp1*tmp1)
  
    if j == 0 and sigma == 0:
        tmp = 0
    elif j == 0 and sigma != 0:
        tmp = (tmp2*(-tmp1)*(-(x[0]-c)/(sigma**2)))
    elif j == 1 and sigma == 0:
        tmp=0
    elif j == 1 and sigma != 0:
        tmp = (tmp2*(-tmp1)*(-1/sigma))
    return tmp

def update_de_do(mynet):
    for k in range(0,mynet.no):
        s = 0
        for i in range(mynet.ni, mynet.ni+mynet.ni*mynet.mf):
            for j in range(0,2):
                do_dp = dmf_dp(mynet, i, j, k)
                if np.isnan(do_dp):
                    print('problem in update') 
                if np.isnan(mynet.de_do[i, k]):
                    print('problem in other chain derivative')
                mynet.mparam_de_do[s,j, k] = mynet.mparam_de_do[s,j, k] + mynet.de_do[i, k]*do_dp
            s = s + 1
    return mynet

def update_parameter(mynet, step_size):
    tmp = mynet.mparam_de_do
    tmp = np.multiply(tmp,tmp)
    len = np.sqrt(np.sum(tmp))
    if len == 0:
        print('prob in update_param')
        sys.exit()
    mynet.mparams = mynet.mparams - step_size * mynet.mparam_de_do/len
    return mynet

def check_decrease_ss(error_array, last_change, current):
    if (current - last_change < 4):
        sts = False
    elif ((error_array[current]   < error_array[current - 1]) and \
            (error_array[current - 1] > error_array[current - 2]) and \
            (error_array[current - 2] < error_array[current - 3]) and \
            (error_array[current - 3] > error_array[current - 4])):
        sts = True
    else:
        sts = False
    return sts

def check_increase_ss(error_array, last_change, current):
    if (current - last_change < 4):
        sts = False
    elif ((error_array[current]     < error_array[current - 1]) and \
            (error_array[current - 1] < error_array[current - 2]) and \
            (error_array[current - 2] < error_array[current - 3]) and \
            (error_array[current - 3] < error_array[current - 4])):
        sts = True
    else:
        sts = False
    return sts

def update_step_size(mynet, RMSE, iter, step_size, decrease_rate, increase_rate):
    if check_decrease_ss(RMSE, mynet.last_decrease_ss, iter):
        step_size = step_size*decrease_rate		
        mynet.last_decrease_ss = iter
    elif check_increase_ss(RMSE, mynet.last_increase_ss, iter):
        step_size = step_size*increase_rate
        mynet.last_increase_ss = iter
    return [mynet, step_size]

def run_anfis(data, idx, epoch_n, mf, step_size, decrease_rate, increase_rate):
    ## DIVIDE DATA AS INPUT AND OUTPUT
    inputs = data[:,0:idx]
    outputs = data[:,idx:]
    ndata = np.size(data,0)         # No. of training samples/data (i.e. no. of rows)
    ni = np.size(inputs,1)
    no = np.size(outputs,1)
    ## DEFINE MINIMUM & MAXIMUM OF INPUTS TO DETERMINE INITIAL MEMBERSHIP FUNCTION AND SOME OTHER VARIABLES
    mn = [np.min(inputs[:,x]) for x in range(ni)]
    mx = [np.max(inputs[:,x]) for x in range(ni)]
    mm = np.subtract(mx,mn)

    nc = mf
    Node_n = ni + ni*mf + 3*nc + 1
    min_RMSE = np.inf
    mparams2 = np.array([])

    for i in range(ni):
        tmp  = np.linspace(mn[i], mx[i], num = mf)
        tmp.shape = (mf,1)
        tmp = np.concatenate((np.matlib.repmat(mm[i]/6,mf,1),tmp), axis = 1)
        if i == 0:
            mparams2 = tmp
        else:
            mparams2 = np.concatenate((mparams2,tmp), axis = 0)
    
    mparams = np.zeros((ni*mf,2,no))

    for i in range(no):
        mparams[:,:,i] = mparams2

    kparams = np.zeros((nc,ni+1,no))        # define initial kalman parameters with all zeros

    ## CREATE CONNECTION MATRIX AND NODES ARRAY
    # connection matrix show which node connect to another
    # nodes vector shows the output of certain node
    config = np.zeros((Node_n,Node_n))

    # <<<<for CANFIS original code changed <!--nodes=zeros(Node_n,1);-->. >>>>
    nodes = np.zeros((Node_n,no))

    # inputs - layer1 connections
    st = ni
    for i in range(ni):
        for j in range(mf):
            config[i,st+j] = 1
        st = st + mf

    # layer1-layer2 connections
    st = ni + ni*mf
    if np.size(inputs,1) == 2:
        for i in range(mf):
            config[ni+i,st] = 1
            config[ni+mf+i,st] = 1
            st = st + 1            
    elif np.size(inputs,1) == 3:
        for i in range(mf):
            config[ni+i,st] = 1
            config[ni+mf+i,st] = 1
            config[ni+2*mf+i,st] = 1
            st = st + 1
    elif np.size(inputs,1) == 4:
        for i in range(mf):
            config[ni+i,st] = 1
            config[ni+mf+i,st] = 1
            config[ni+2*mf+i,st] = 1
            config[ni+3*mf+i,st] = 1
            st = st + 1
    elif np.size(inputs,1) == 5:
        for i in range(mf):
            config[ni+i,st] = 1
            config[ni+mf+i,st] = 1
            config[ni+2*mf+i,st] = 1
            config[ni+3*mf+i,st] = 1
            config[ni+4*mf+i,st] = 1
            st = st+1
    elif np.size(inputs,1)==6:
        for i in range(mf):
            config[ni+i,st] = 1
            config[ni+mf+i,st] = 1
            config[ni+2*mf+i,st] = 1
            config[ni+3*mf+i,st] = 1
            config[ni+4*mf+i,st] = 1
            config[ni+5*mf+i,st] = 1
            st = st+1
    else:
        exit()

    # layer2-layer3 connections
    for i in range(nc):
        for j in range(nc):
            config[ni+ni*mf+i,ni+ni*mf+nc+j] = 1

    # layer3-layer4 connections
    for i in range(nc):
        config[ni+ni*mf+nc+i,ni+ni*mf+2*nc+i] = 1

    # layer4-layer5 connections
    for i in range(nc):
        config[ni+ni*mf+2*nc+i,-1] = 1

    # inputs - layer4  connections
    for i in range(ni):
        for j in range(nc):
            config[i,ni+ni*mf+2*nc+j] = 1

    ## CREATE A NETWORK ARCHITECTURE
    mynet = Architecture(config, mparams, kparams, nodes, ni, no, mf, nc)

    ## ITERATION LOOP
    RMSE = np.zeros(epoch_n)
    for iter in range(epoch_n):
        layer_1_to_3_output = np.zeros((Node_n,mynet.no,ndata))
        anfis_output = np.zeros((ndata,mynet.no))
        target = np.zeros(mynet.no)
        de_dout = np.zeros(mynet.no)
        for j in range(ndata):
            # set j th input into the networks
            # <<<<for CANFIS original code changed <!--mynet.nodes(1:mynet.ni)=inputs(j,:)';-->. >>>>
            for k in range(mynet.no):
                mynet.nodes[0:mynet.ni,k] = np.transpose(inputs[j,:]) ######################################################

            # get node outputs from layer 1 to layer 3
            mynet = calculate_output1(mynet)
            mynet = calculate_output2(mynet)
            mynet = calculate_output3(mynet)

            # save outputs of layer 1 to 3
            for k in range(mynet.no):
                layer_1_to_3_output[:,k,j] = mynet.nodes[:,k]

            # calculate kalman params
            kalman_data = get_kalman_data(mynet,outputs[j,:])
            # update kalman params
            mynet = mykalman(mynet,kalman_data,j)


        # clear all derivatives as zero
        mynet = clear_de_dp(mynet)

        for j in range(ndata):
            # get output of layer 1 to 3 from layer_1_to_3_output to avoid recalculation of layer1-2-3
            for k in range(mynet.no):
                mynet.nodes[:,k] = layer_1_to_3_output[:,k,j]
            
            # calculate outputs of layer 4
            mynet = calculate_output4(mynet)

            # calculate outputs of layer 5
            mynet = calculate_output5(mynet)
            # calculate network output
            for k in range(mynet.no):
                anfis_output[j,k] = mynet.nodes[-1,k]
                target[k] = outputs[j,k]
                # calculate differential of error
                de_dout[k] = -2*(target[k] - anfis_output[j,k])
    
            # backpropagete errors
            mynet = calculate_de_do(mynet,de_dout)
            mynet = update_de_do(mynet)    

        # calculate one train loop error
        diff = anfis_output - outputs
        total_squared_error = np.sum(diff*diff)
        RMSE[iter] = np.sqrt(np.sum(total_squared_error)/(ndata*mynet.no))
        print(str(iter) + '...aggregate rmse error is :\n' + str(RMSE[iter]))
        if RMSE[iter] < min_RMSE:
           bestnet = mynet
           min_RMSE = RMSE[iter]
        
        # update membership parameter
        mynet = update_parameter(mynet, step_size)

        # update step size
        (mynet,step_size) = update_step_size(mynet,RMSE,iter,step_size,decrease_rate, increase_rate)

    ## CALCULATE BEST NETS OUTPUT
    mynet = bestnet

    for j in range(ndata):
        for k in range(mynet.no):
            mynet.nodes[0:mynet.ni,k] = np.transpose(inputs[j,:])
        mynet = calculate_output1(mynet)
        mynet = calculate_output2(mynet)
        mynet = calculate_output3(mynet)
        mynet = calculate_output4(mynet)
        mynet = calculate_output5(mynet)
        for k in range(mynet.no):
            anfis_output[j,k] = mynet.nodes[-1,k]