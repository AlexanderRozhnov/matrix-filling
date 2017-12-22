import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.sparse.linalg import norm
from copy import deepcopy

class ALS(object):
    
    def __init__(self, k=10, lambda_ = 10, max_iter=10, tol=1e-5, missed_val = 'NaN', print_iter=False):
        self._k = k
        self._lambda = lambda_
        self._max_iter = max_iter
        self._missed_val = missed_val
        self._tol = tol
        self._print_iter = print_iter
        self._errors = []
    
    def fit(self, ratings):
        #initial assigning of factors X, Y
        self.data = ratings
        self.nonzero = ratings.nonzero()
        X, Y = np.abs(np.random.rand(self._k, ratings.shape[0])), np.abs(np.random.rand(self._k, ratings.shape[1]))
        self.X = X
        self.Y = Y            
        self.bias = np.array([])
            
        for i in range(self._max_iter):
            column_diff_norm = 0
            if self._print_iter:
                print(i)

            #flag shows that X and Y changed much, so we should not stop
            flag = True
            
            self.update_proj()
            self.bias = np.append(self.bias, norm(self.data - self.proj, ord='fro'))

            for j in range(ratings.shape[0]):
                #take row in matrix of observations with which we will work
                row = ratings.getrow(j).toarray().ravel()

                #find arguments of elemnts in row which were not missed
                args = []
                if self._missed_val == 'NaN':
                    args = np.argwhere(~np.isnan(row)).T[0]
                else:
                    args = np.argwhere(row != self._missed_val).T[0]

                #create matrices to save temporary results
                summation_inv = np.zeros((self._k, self._k))
                summation = np.zeros((self._k, 1))

                #for every non-nan element in row we take corresponding column of Y and make manipulations
                for arg in args:
                    summation_inv = summation_inv + (Y[:, arg].reshape(-1,1)).dot(Y[:, arg].reshape(1, -1))
                    summation = summation + row[arg] * Y[:, arg].reshape(-1,1)

                #update the corresponding column of X
                new_X = solve(summation_inv + self._lambda * np.eye(self._k), summation)
                column_diff_norm += np.linalg.norm(X[:,j] - new_X) / np.linalg.norm(X[:,j])
                X[:,j] = new_X.reshape(-1,)
             
            self.X = X
            self.Y = Y
            self.update_proj()
            self.bias = np.append(self.bias, norm(self.data - self.proj, ord='fro'))

            #repeat everything for matrix Y
            for j in range(ratings.shape[1]):
                #take column in matrix of observations with which we will work
                column = ratings.getcol(j).toarray().ravel()

                #find arguments of elemnts in column which were not missed
                args = []
                if self._missed_val == 'NaN':
                    args = np.argwhere(~np.isnan(column)).T[0]
                else:
                    args = np.argwhere(column != self._missed_val).T[0]

                #create matrices to save temporary results
                summation_inv = np.zeros((self._k, self._k))
                summation = np.zeros((self._k, 1))

                #for every non-nan element in row we take corresponding column of Y and make manipulations
                for arg in args:
                    summation_inv = summation_inv + (X[:, arg].reshape(-1,1)).dot(X[:, arg].reshape(1, -1))
                    summation = summation + column[arg] * X[:, arg].reshape(-1,1)

                #update the corresponding column of Y
                new_Y = solve(summation_inv + self._lambda * np.eye(self._k), summation)
                column_diff_norm += np.linalg.norm(Y[:,j] - new_Y) / np.linalg.norm(Y[:,j])
                Y[:,j] = new_Y.reshape(-1)

            self._errors.append(column_diff_norm)
            if column_diff_norm < 1e-5:
                break

        #save the results as the attribute of class
        self._X = X
        self._Y = Y
        
    def update_proj(self):
        proj_data = np.empty(self.nonzero[0].size)
        for i in range(self.nonzero[0].size):
            proj_data[i] = self.X.T[self.nonzero[0][i], :].dot(self.Y[:, self.nonzero[1][i]])
        self.proj = sparse.csr_matrix((proj_data, self.nonzero), self.data.shape)
    
    def get_factors(self):
        return self._X, self._Y
    
    def continue_fit(self, X, Y):
        pass
        
    def predict(self):
        return (self._X.T).dot(self._Y)
    
    def get_errors(self):
        return self._errors
