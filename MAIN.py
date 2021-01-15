import numpy as np
import pandas as pd
import canfis_read_data as cin
import anfis as af

def main():
    input_idx = 6
    max_iter = 100
    mf = 3
    step = 0.001
    inc_rate = 0.9
    dec_rate = 0.1

    [X,Y] = cin.loadData('AI_ML_DL\\Neuro-Fuzzy\\CANFIS_scratch','demo.csv',inputNo = input_idx)    # Change path and filename here
    # X = np.transpose(X)
    # Y = np.transpose(Y)
    X = np.array(X)
    Y = np.array(Y)
    # X = np.array(X[:,0:3])
    # Y = np.array(Y[:,1])
    input_no = len(X[0])
    for j in range(input_no):
        max_X = np.max(X[:,j])
        min_X = np.min(X[:,j])

        for i in range(0,len(X)):
            X[i,j] = round((X[i,j] - min_X)/(max_X - min_X), 4)

    target_no = 1 if Y.ndim == 1 else len(Y[0])
    if Y.ndim == 1:
        max_Y = np.max(Y)
        min_Y = np.min(Y)

        for i in range(0,len(Y)):    
            Y[i] = round((Y[i] - min_Y)/(max_Y - min_Y), 4)
    else:
        for j in range(target_no):
            max_Y = np.max(Y[:,j])
            min_Y = np.min(Y[:,j])

            for i in range(0,len(Y)):    
                Y[i,j] = round((Y[i,j] - min_Y)/(max_Y - min_Y), 4)
    data = np.concatenate((X, Y),axis=1)
    print(data[0])
    af.run_anfis(data, input_idx, max_iter, mf, step, inc_rate, dec_rate)

if __name__ == "__main__":
    main()