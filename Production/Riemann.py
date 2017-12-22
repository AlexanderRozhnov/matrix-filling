import numpy as np
import copy
import scipy.sparse as scp
from scipy.sparse import linalg

class Problem:
    def __init__(self, A, tau, rank, maxiter, accuracy):
        self.tau = tau
        self.rank = rank
        self.maxiter = maxiter
        self.accuracy = accuracy
        
        self.A = A
        self.m = A.shape[0]
        self.n = A.shape[1]

        self.Omega = A.nonzero()

    def projection(self, Y1, Y2):
        rows = self.Omega[0]
        cols = self.Omega[1]
        
        nnz = rows.shape[0]
        data = np.zeros(rows.shape[0])
        
        for it in range(0, nnz):
            data[it] = np.dot(Y1[rows[it]].T, Y2[cols[it]])
        
        proj = scp.csr_matrix((data, (rows, cols)), shape=self.A.shape)
        
        return proj
    
    def grad(self, svd, Rw):
        U = svd[0]
        V = svd[2]
        Ru = Rw.T.dot(U)
        Rv = Rw.dot(V)
        
        M = np.dot(U.T, Rv)
        
        Up = Rv - np.dot(U, M)
        Vp = Ru - np.dot(V, M.T)
        
        grad_value = U.dot(M.dot(V.T)) + Up.dot(V.T) + U.dot(Vp.T)
        grad_list = [M, Up, Vp]
        grad = [grad_value, grad_list]
        
        return grad
    
    def create_Xw(self, X_svd):
        Y1 = np.dot(X_svd[0], X_svd[1])
        Y2 = X_svd[2]
        Xw = self.projection(Y1, Y2)
        return(Xw)
    
    def compute_initial_guess(self, X_new_svd, Rw, grad):
        """
        Compute the initial guess for line search t* = argmin_t f(X+t*eta)
        Args:
            X_new_svd (list): SVD in the new point in the form [U, S, V], X = USV.T
            Rw (np.array): Rw = Xw - Aw
            eta_new (list): conjugate vector in the new point in the form [value, [M, Up, Vp]]
        Returns:
            t (float)
        
        """
        R = -Rw
        U = X_new_svd[0]
        V = X_new_svd[2]
        M = grad[1][0]
        Up = grad[1][1]
        Vp = grad[1][2]
        
        Y1 = np.dot(U, M) + Up
        Y1 = np.hstack((Y1, U))
        Y2 = np.hstack((V, Vp))
        N = self.projection(Y1, Y2)
        top = N.T.dot(R)
        bottom = N.T.dot(N)
        t = top.diagonal().sum() / bottom.diagonal().sum()
        return t
    
    def compute_retraction(self, X_new_svd, tangent):

        k = self.rank
        M = tangent[0]
        Up = tangent[1]
        Vp = tangent[2]

        U  = X_new_svd[0]
        Sigma = X_new_svd[1]
        V  = X_new_svd[2]
        
        Zero = np.zeros((k, k))
        (Qu, Ru) = np.linalg.qr(Up)
        (Qv, Rv) = np.linalg.qr(Vp)
        top = np.hstack((Sigma + M, Rv.T))
        bottom = np.hstack((Ru, Zero))
        S = np.vstack((top, bottom))
        Us, Sigmas, Vs = np.linalg.svd(S)
        
        Vs = Vs.T
        Sigma_plus = np.diag(Sigmas[0:k] + 1e-16)
        
        Current = np.hstack((U, Qu))
        U_plus = np.dot(Current, Us[:, :k])

        Current = np.hstack((V, Qv))
        V_plus = np.dot(Current, Vs[:, :k])

        retraction = [U_plus, Sigma_plus, V_plus]
        return retraction
    
    def Initialization(self):

        m = self.m
        n = self.n
        k = self.rank
        
        diag = np.random.uniform(0, 1, k)
        diag = np.sort(diag)[::-1]
        
        S1 = np.diag(diag)
        U1 = np.random.uniform(0, 1, (m, k))
        V1 = np.random.uniform(0, 1, (n, k))
        
        Qu1, Ru1 = np.linalg.qr(U1)
        Qv1, Rv1 = np.linalg.qr(V1)
        X = [Qu1, S1, Qv1]
        
        Xw = self.create_Xw(X)
        R = Xw - self.A
            
        # Compute the gradient and direction
        xi = self.grad(X, R)
        eta = copy.deepcopy(xi)
            
        # Compute Retraction
        X_new = self.line_search(X, R, xi, eta)

        return(X, X_new, xi, eta)

    
    def vector_transport(self, X_old_svd, X_new_svd, v_list):
        """
        Calculate transport vector from previous tangent space to new tangent space
        
        Args:
            X_old_svd (list): SVD in the previous point in the form [U, S, V], X = USV.T
            X_new_svd (list): SVD in the new point in the form [U+, S+, V+], X+ = U+S+V+.T
            v_list (list): vector, we need to transport in the form [M, Up, Vp]

        Returns:
            np.array, list: Transport vector and corresponding matrices [value, [M+, Up+, Vp+]]
        """
        U = X_old_svd[0]
        S = X_old_svd[1]
        V = X_old_svd[2]
    
        U_plus = X_new_svd[0]
        S_plus = X_new_svd[1]
        V_plus = X_new_svd[2]
        
        M = v_list[0]
        Up = v_list[1]
        Vp = v_list[2]
        
        Av = np.dot(V.T, V_plus)
        Au = np.dot(U.T, U_plus)
        
        Bv = np.dot(Vp.T, V_plus)
        Bu = np.dot(Up.T, U_plus)
        
        M_plus_one = np.dot(Au.T, np.dot(M, Av))
        U_plus_one = np.dot(U, np.dot(M, Av))
        V_plus_one = np.dot(V, np.dot(M.T, Au))
        
        M_plus_two = np.dot(Bu.T, Av)
        U_plus_two = np.dot(Up, Av)
        V_plus_two = np.dot(V, Bu)
        
        M_plus_three = np.dot(Au.T, Bv)
        U_plus_three = np.dot(U, Bv)
        V_plus_three = np.dot(Vp, Au)
        
        M_plus = M_plus_one + M_plus_two + M_plus_three
        Up_plus = U_plus_one + U_plus_two + U_plus_three
        Up_plus = Up_plus - np.dot(U_plus, np.dot(U_plus.T, Up_plus))
        
        Vp_plus = V_plus_one + V_plus_two + V_plus_three
        Vp_plus = Vp_plus - np.dot(V_plus, np.dot(V_plus.T, Vp_plus))
        
        transport_value = np.dot(U_plus, np.dot(M_plus, V_plus.T)) + np.dot(Up_plus, V_plus.T) \
                                                        + np.dot(U_plus, Vp_plus.T)
        
        transport_list = [M_plus, Up_plus, Vp_plus]
        transport = [transport_value, transport_list]
        return transport
    
    
    def conjugate_direction(self, X_old_svd, X_new_svd, xi_old, xi_new, eta_old):
        """
        Compute the conjugate direction by PR+ in the new point
        
        Args:
            X_old_svd (list): SVD in the previous point in the form [U, S, V], X = USV.T
            X_new_svd (list): SVD in the new point in the form [U+, S+, V+], X+ = U+S+V+.T
            xi_old (list): tangent vector in the previous point in the form [value, [M, Up, Vp]]
            xi_new (list): tangent vector in the new point in the form [value, [M, Up, Vp]]
            eta_old (list): conjugate vector in the previous point in the form [value, [M, Up, Vp]]

        Returns:
            np.array, list: conjugate vector and corresponding list in the new point [M+, Up+, Vp+]
        """
        # Transport previous gradient and direction to current tangent space:
        xi_bar = self.vector_transport(X_old_svd, X_new_svd, xi_old[1])
        eta_bar = self.vector_transport(X_old_svd, X_new_svd, eta_old[1])
        
        # Compute conjugate direction
        delta = xi_new[0] - xi_bar[0]
        top = np.trace(np.dot(delta.T, xi_new[0]))
        
        # Пока с trace, переделать
        bottom = np.trace(np.dot(xi_old[0].T, xi_old[0]))
        betta = np.maximum(0, top/bottom)
        #print('betta', betta)
        eta_value = -xi_new[0] + betta*eta_bar[0]
        
        # Renew eta_list
        M_eta_bar = eta_bar[1][0]
        Up_eta_bar = eta_bar[1][1]
        Vp_eta_bar = eta_bar[1][2]
        
        M_xi = xi_new[1][0]
        Up_xi = xi_new[1][1]
        Vp_xi = xi_new[1][2]
        

        M_eta = -M_xi + betta*M_eta_bar
        Up_eta = -Up_xi + betta*Up_eta_bar
        Vp_eta = -Vp_xi + betta*Vp_eta_bar
        
        eta_list = [M_eta, Up_eta, Vp_eta]
        
        eta = [eta_value, eta_list]
        
        # Compute angle between conjugate direction and gradient:
        
        top = np.trace(np.dot(eta[0].T, xi_new[0]))
        bottom = np.sqrt(np.trace(np.dot(eta[0].T, eta[0]))*np.trace(np.dot(xi_new[0].T, xi_new[0])))
        alpha = top/bottom
        #print(bottom)
        
        # Reset to gradient if desired:
        if np.abs(alpha) <= 0.1:
            #print('alpha: ', alpha)
            eta_value = xi_new[0].copy()
            eta_list = xi_new[1].copy()
            eta = [eta_value, eta_list]
        
        return eta
    
    
    def line_search(self, X, Rw, grad, con_grad):
        """
        Line search for finding m and t
        Args:
            
        Returns:
        """
        f_old = 0.5*(linalg.norm(Rw)**2)
        t = self.compute_initial_guess(X, Rw, con_grad)    

        m = 0        
        while True:
            step = (0.5**m)*t

            M_step = con_grad[1][0]*step
            
            Up_step = con_grad[1][1]*step
            Vp_step = con_grad[1][2]*step
            eta_step = [M_step, Up_step, Vp_step]
            
            # Compute Retraction
            X_new = self.compute_retraction(X, eta_step)

            f_new = 0.5*(linalg.norm(self.create_Xw(X_new) - self.A)**2)
            #print('norm', np.linalg.norm(antigrad[0]))
            #print('f_old - f_new ', f_old - f_new)
            #print('current m ', m)
            #print('step', step)
            
            if (f_old - f_new >= -0.0001*step*np.trace(np.dot(grad[0].T, con_grad[0]))):
                break

            m = m + 1

        return X_new
    
    def LRGeomCG(self):
        
        residuals = []
        X_old, X_new, xi_old, eta_old = self.Initialization()
        iters=0
        self.bias = np.array([])
        while True:
            if(iters > self.maxiter):
                break
            iters+=1
            #print('Num_of_iter ', iters)
            Xw = self.create_Xw(X_new)
            R = Xw - self.A
            self.bias = np.append(self.bias, linalg.norm(R, ord='fro'))
            xi_new = self.grad(X_new, R)
            
            if np.linalg.norm(xi_new[0]) <= self.tau:
                break
            
            # Compute a conjugate direction by PR+
            eta_new = self.conjugate_direction(X_old, X_new, xi_old, xi_new, eta_old)
            # Compute Retraction
            X_cur = self.line_search(X_new, R, xi_new, eta_new)
            
            # Renew X_old_svd and X_new_svd
            X_old = copy.deepcopy(X_new)
            X_new = copy.deepcopy(X_cur)
            
            # Renew xi_old and eta_new
            xi_old = copy.deepcopy(xi_new)
            eta_old = copy.deepcopy(eta_new)

            # Calculate residuals
            Xw = self.create_Xw(X_new)
            R = Xw - self.A
            cur_res = 0.5*(linalg.norm(R, ord='fro')**2)
            residuals.append(cur_res)
            if cur_res <= self.accuracy:
                break
       
        self.residuals = residuals
        self.approx = copy.deepcopy(X_new)
        pass
