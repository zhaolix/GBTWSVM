import numpy as np
import scipy.optimize as optimize
from tools.gen_ball import gen_balls
from tools.addNoisy import recreat_data
from tools.devitedata import Dive_Data
from tools.Evaluation import evaluation
import time



class GBTWSVM():
    def __init__(self, c_train, r_train, y_train, c1 = 1, c2 = 1): 
        self.c_train = c_train
        self.r_train = r_train
        self.y_train = y_train
        # Twin SVM settings
        self.c1 = c1  
        self.c2 = c2  

    def minimize(self, alpha, R, P):
        '''
        Solve quadratic problem using SciPy optimize
        '''
        g = -(alpha.T @ R) + 0.5 * alpha.T @ P @ alpha
        return g
    
    def train(self):
        '''
        Solve for w, b
        '''
        c_train = self.c_train
        r_train = self.r_train
        y_train = self.y_train
        if np.array(y_train).ndim == 1:
            y_train = np.array([y_train]).T 
        rA, cA = np.array(r_train)[y_train[:, 0] == 1],np.array(c_train)[y_train[:, 0] == 1]   # 取得正样本数据
        rB, cB = np.array(r_train)[y_train[:, 0] == -1],np.array(c_train)[y_train[:, 0] == -1]  # 取得负样本数据     
        # Number of positive and negative samples
        m1 = cA.shape[0] 
        m2 = cB.shape[0]
        # Create an (m1,1) matrix where all the elements are 1
        e1 = np.ones((m1, 1))
        e2 = np.ones((m2, 1))
        # GBTWSVM 1
        E = np.hstack((cA, e1))
        F = np.hstack((cB, e2))
        P1 = F @ np.linalg.pinv(E.T @ E) @ F.T 
        R1 = (rB + e2.T).T
        # GBTWSVM 2
        R = np.hstack((cA, e1))
        S = np.hstack((cB, e2))
        P2 = R @ np.linalg.pinv(S.T @ S) @ R.T 
        R2 = (rA + e1.T).T
        # Initialize alpha and gamma
        alpha0 = np.zeros((np.size(F, 0), 1)).ravel() 
        gamma0 = np.zeros((np.size(R, 0), 1)).ravel()
        # Scipy optimize
        # Define the boundary range of the parameter
        b1 = optimize.Bounds(0, self.c1)
        b2 = optimize.Bounds(0, self.c2)
        alpha = optimize.minimize(self.minimize, x0=alpha0, args=(R1, P1), method='L-BFGS-B', bounds=b1).x 
        gamma = optimize.minimize(self.minimize, x0=gamma0, args=(R2, P2), method='L-BFGS-B', bounds=b2).x
        if alpha.ndim == 1:
            alpha = np.array([alpha]).T
        if gamma.ndim == 1:
            gamma = np.array([gamma]).T
        # Solve parameter, z1=[w1,b1]^T, z2=[w2,b2]^T
        epsilon = 1e-16
        I = np.eye(len(E.T @ E))
        self.u = -np.linalg.pinv(E.T @ E + epsilon * I) @ F.T @ alpha
        I = np.eye(len(S.T @ S))
        self.v = np.linalg.pinv(S.T @ S + epsilon * I) @ R.T @ gamma
        return

    def predict(self, X_test):
        w1 = self.u[:-1]   
        b1 = self.u[-1:]   
        w2 = self.v[:-1]   
        b2 = self.v[-1:]   
        # Get two non-parallel hyperplanes
        surface1 = X_test @ w1 + b1
        surface2 = X_test @ w2 + b2
        dist1 = abs(surface1)  # class 1
        dist2 = abs(surface2)  # class -1
        # Determine the sample point category
        y_hat = np.argmax(np.hstack((dist1, dist2)), axis=1) 
        if y_hat.ndim == 1:
            y_hat = np.array([y_hat]).T
        y_hat = np.where(y_hat == 0, -1, y_hat) 
        return y_hat
    
if __name__ == '__main__':
    datae = ["Australian", "Breast", "Diabetes", "Fourclass", "German.numer", "Heart", "Ionosphere", "Liver-disorders", "Spambase", "Splice", "WDBC"]
    for i in range(len(datae)):
        urlz = r"./data/" + datae[i] + ".csv" 
        name = datae[i]
        nor = True 
        pr = 0.2  
        # Add noise
        for j in range(0,11): 
            Noisy = 0.01 * j
            gbtwsvm_total_time = 0
            gbtwsvm_num_total = 0
            gbtwsvm_eva_total=np.array([0,0,0,0]).astype(np.float64)
            for k in range(1,11):
                gbtwsvm_eva = np.array([0,0,0,0]).astype(np.float64)
                gbtwsvm_num_k = 0
                train, test = Dive_Data(urlz, nor, pr) 
                train = np.array(train)
                test = np.array(test)
                N_data = recreat_data(train, Noisy)
                N_data = np.array(N_data)
                X_train, Y_train = N_data[:,:-1], N_data[:,-1]
                X_test, Y_test = test[:,:-1], test[:,-1]
                pur = 0
                # Generating granular-balls
                for l in range(0,21): 
                    pur = 1 - 0.01 * l
                    for m in range(2, 3):
                        num = m * 1
                        datab = gen_balls(N_data, pur=pur, delbals=num)  # datab [c, r, label]
                        print("Purity:", pur, "The number of granular-balls:", len(datab))
                        c,r,y = [],[],[]
                        for ii in datab:
                            c.append(ii[0])
                            r.append(ii[1])
                            y.append(ii[-1])
                        # Model calculation
                        gbtwsvm_start_time = time.time()
                        gbtwsvm = GBTWSVM(c,r,y)
                        gbtwsvm.train()
                        y_hat = gbtwsvm.predict(X_test)
                        gbtwsvm_end_time = time.time()
                        eva= np.array(evaluation(Y_test, y_hat))  # (accuracy, precision, recall, f1-score)                    
                        if eva[0] > gbtwsvm_eva[0]:
                            gbtwsvm_eva = eva
                            gbtwsvm_num_k = len(datab)
                        gbtwsvm_iteration_time = gbtwsvm_end_time - gbtwsvm_start_time
                        gbtwsvm_total_time += gbtwsvm_iteration_time
                print("Dataset:", name, "Noisy:", str(Noisy), "Cycle:", k, "The number of granular-balls:", gbtwsvm_num_k,"Evaluation index:", gbtwsvm_eva)
                gbtwsvm_eva_total += gbtwsvm_eva
                gbtwsvm_num_total += gbtwsvm_num_k
            gbtwsvm_num_av = gbtwsvm_num_total/k
            gbtwsvm_eva_av = gbtwsvm_eva_total/k
            gbtwsvm_average_time = gbtwsvm_total_time / (k*(l+1))
            print("Dataset:", name, "Noisy:", str(Noisy), "The number of granular-balls:", gbtwsvm_num_av, "Evaluation index:", gbtwsvm_eva_av, "Time:", gbtwsvm_average_time)