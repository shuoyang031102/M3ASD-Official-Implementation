import scipy.io
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys


import numpy as np
from numpy import *
from scipy.io import loadmat, savemat
import scipy.sparse as SP
import matplotlib.pyplot as plt

import scipy.sparse as sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import time
import scipy.sparse as sps

from latent_lrr import *

sys.path.append('../utils')
from abide import get_original_data_with_site_name
from pairwisedistances import get_distance
from pairwisedistances import get_pairwise_distances
from config import CONFIG



def singular_value_shrink(A, alpha):
    A[np.isnan(A)] = 0
    A[np.isinf(A)] = 1
    U, lamb, VT = np.linalg.svd(A)
    idx = np.where(lamb > alpha)[0]
    if len(idx) == 0:
        return np.zeros_like(A)
    elif len(idx) == 1:
        return np.outer(U[:,0], (lamb[0] - alpha) * VT[0,:])
    else:
        diags = np.maximum(lamb[idx] - alpha,0)
        return U[:,idx] @ SP.diags(diags).dot(VT[idx,:])


def solve_l2(w:np.array, alpha):

    nw = np.linalg.norm(w)
    if nw > alpha:
        return (nw - alpha) / nw * w
    else:
        return np.zeros_like(w)

def solve_l21(W:np.array, alpha):

    E = np.zeros_like(W)
    m, n = W.shape
    for i in range(n):
        E[:,i] = solve_l2(W[:,i], alpha)
    return E

def __solve_low_rank_representation(X:np.array, A:np.array, lamb):

    print(X.shape, A.shape)

    tol = 1e-8
    maxIter = 1e6
    d, n = X.shape
    d, m = A.shape
    c = 1e-3
    maxc = 1e10
    rho = 1.2

    # Initialize primal variables
    J = np.zeros(shape=[m,n])
    Z = np.zeros(shape=[m,n])
    # E = np.zeros(shape=[d,n])
    E = np.random.randn(d,n)

    # Initialize dual variables
    Y1 = np.zeros(shape=[d,n])
    Y2 = np.zeros(shape=[m,n])

    # Some data that will be used many times
    inv_atapi = np.linalg.inv(A.T @ A + np.eye(m))
    atx = A.T @ X

    iter = 0
    convergence = False
    ranks = []
    while not convergence:
        iter += 1

        # Update primal variabble J
        J = singular_value_shrink(Z + Y2 / c, 1 / c) #奇异值收缩

        # Update primal variable Z
        Z = inv_atapi @ (atx - A.T@E + J + (A.T @ Y1 - Y2)/c)

        # Update primal variable E
        xmaz = X - A @ Z
        E = solve_l21(xmaz + Y1 / c, alpha=lamb/c)

        # Update dual variable Y1 and Y2
        leq1 = xmaz - E # linear equality 1
        leq2 = Z - J    # linear equality 2
        Y1 += c * leq1
        Y2 += c * leq2

        ranks.append(np.linalg.matrix_rank(Z,tol=1e-3*np.linalg.norm(Z, 2)))
        if iter % 10 == 0:
            print("Iteration %d, c:%.6f, Rank:%d, Equality 1 violation: %.5f, Equality 2 violation: %.5f"
                  %
                  (iter, c, np.linalg.matrix_rank(Z,tol=1e-3*np.linalg.norm(Z, 2)), np.max(np.abs(leq1)), np.max(np.abs(leq2)))
            )
        c = min(c * rho, maxc)

        if max(np.max(np.abs(leq1)), np.max(np.abs(leq2))) < tol:
            convergence = True

    return Z, E, ranks



def solve_low_rank_representation(X, lamb):

    # Q = orth(X.T)
    # A = X @ Q
    Z, E, ranks = __solve_low_rank_representation(X, X, lamb)


    return X @ Z, E, ranks,Z


def __solve_low_rank_sparse_graph_LRR_representation(X:np.array, W:np.array, lamb, garma = 1.9, beta = 1.1, rho=1.3, DEBUG = 0):




    DCol = W + 2
    m,n=W.shape
    D = sparse.spdiags(DCol, np.arange(0, m, 1),m, n)
    L = D - W

    normfX =  np.linalg.norm(X)
    tol1 = 1e-4
    #约束中的错误阈值
    tol2 = 1e-5
    #解决方案变化的阈值
    d,n = X.shape
    maxIter = 500
    max_mu = 1e5
    norm2X = np.linalg.norm(X, ord=2)
    #二范数
    mu = np.dot(min(d, n), 1e-5)
    inv_a = np.linalg.inv(X.T @ X+ np.eye(m))
    inv_b = np.linalg.inv(X @ X.T + np.eye(d));
    atx = np.dot(X.T, X)
    eta = np.dot(norm2X, norm2X)* 1.001
    # eta需要大于| X | 2^2，但不需要太大
    E = np.zeros((d, n))
    #0填充
    Y1 = np.zeros((d, n))
    Y2 = np.zeros((n, n))
    Y3 = np.zeros((d, d))
    Z = np.eye(n, n)
    #生成对角阵
    J = np.zeros((n, n))
    XZ = np.zeros((d, n))
    sv = 5
    svp = sv
    convergenced = False
    iter = 0

    ranks =[]
    while not convergenced and iter < maxIter:
        iter = iter + 1;
        # J = singular_value_shrink(J, 1 / mu)
        Ek = E
        Zk = Z
        Jk = J

        XZ = np.dot(X,Z)
        ZLT = np.dot(Z,np.transpose(L))
        ZL = np.dot(Z,L)

        M = np.dot(beta, ZLT+ZL)
        M = M + np.dot(np.dot(mu, np.transpose(X)), (XZ - X + E - np.divide(Y1,mu)))
        M = M + np.dot(mu, (Z - J + np.divide(Y2, mu)))
        M = Z - np.divide(M , eta)
        M[np.isnan(M)] = 0
        M[np.isinf(M)] = 1
        U, S, V = np.linalg.svd(M, 'econ')
        #总共有三个返回值u,s,v
       #u大小为(M,M)，s大小为(M,N)，v大小为(N,N)
        S = np.diag(S)
        svp = len(np.argwhere(S > 1 / (np.dot(mu, eta))))

        if svp < sv:
            sv = min(svp + 1, n)
        else:
            sv = min(svp + round(0.05 * n), n)

        if svp >= 1:

            #svp = sv
            S = S[:svp] - 1 / (np.dot(mu, eta))
        else:
            svp = 1
            S = np.zeros((1,1))
        AU = U[:, : svp]
        As = S
        AV = V[:, : svp]
        Z = np.diag(np.diag(As))
        Z = np.dot(AU, Z)
        Z = np.dot(Z, np.transpose(AV))
        S=Z
        Z = inv_a @ (atx - X.T @ E + S + (X.T @ Y1 - Y2) / mu)



        #L = (X.T @ (X - X @ Z - E)   + S + (Y1 @ X.T - Y3) / mu) @ inv_b
        # Z = inv_a @ (atx - X @ L @ X.T - X.T @ E + J + (X.T @ Y1 - Y2) / mu)
        # L = ((X - X @ Z - E) @ X.T + S + (Y1 @ X.T - Y3) / mu) @ inv_b
        # udpate L

        #Z = np.dot(np.dot(AU, np.diag(np.diag(As))), np.transpose(AV))
        XZ = X @ Z
        temp = Z + np.divide(Y2 , mu)
        te1=temp - lamb / mu
        te1[te1 < 0] = 0
        te2=temp + lamb / mu
        te2[te2 > 0] = 0
        J = te1 + te2
        # J[J < 0] = 0


        temp = X - XZ
        temp = temp + np.divide(Y1, mu)
        te1 = temp - lamb / mu
        te1[te1 < 0] = 0
        te2=temp + lamb / mu
        te2[te2 > 0] = 0
        E = te1 + te2
        relChgZ = np.divide(np.linalg.norm(Zk - Z), normfX)
        relChgE = np.divide(np.linalg.norm(E - Ek), normfX)
        relChgJ = np.divide(np.linalg.norm(J - Jk), normfX)
        relChg = max(max(relChgZ, relChgE), relChgJ)





        dY1 = X - XZ - E
        recErr1 = np.divide(np.linalg.norm(dY1), normfX)
        dY2 = Z - J
        recErr2 = np.divide(np.linalg.norm(dY2), normfX)
        recErr = max(recErr1, recErr2)

        leq1 = X - XZ - E
        leq2 = Z - J

        max_l1 = np.max(np.max(np.abs(leq1)))
        max_l2 = np.max(np.max(np.abs(leq2)))
        stopC1 = max(max_l1, max_l2)


        if stopC1 < tol1 :
            convergenced = True
            break

        if (recErr < tol1) and (relChg < tol2):
            convergenced = True
            break
        ranks.append(np.linalg.matrix_rank(Z, tol=1e-3 * np.linalg.norm(Z, 2)))
        if iter % 10 == 0:
            print("Iteration %d, mu:%.6f, Rank:%d, Equality 1 violation: %.5f, Equality 2 violation: %.5f,Equality  violation: %.5f"
                  %
                  (iter, mu, np.linalg.matrix_rank(Z,tol=1e-3*np.linalg.norm(Z, 2)), np.max(np.abs(recErr)), np.max(np.abs(relChg)),stopC1)
            )

        mu = min(max_mu, mu * rho)
        if not convergenced:
            Y1 = Y1 + np.dot(mu, dY1)
            Y2 = Y2 + np.dot(mu, dY2)
            if mu * relChg < tol2:
                mu = min(max_mu, mu * rho)


    return Z, E, ranks

def solve_low_rank_sparse_graph_LRR_representation(X, lamb):
    """

    :param X: Of shape [d, n], d is data dimensions, n is number of data
    :param lamb:
    :return:
    """
    # Q = orth(X.T)
    # A = X @ Q

    '''matrix_minmax = MinMaxScaler().fit_transform(X)
    cosine_similarity(matrix_minmax)
    pairwise_distances(matrix_minmax, metric='cosine')
    n,m=matrix_minmax.shape
    num=min(n,m)
    X2=matrix_minmax[:num,:num]
    print(X2.shape)'''

    a=X.T

    dis = []
    for i in range(a.shape[0] - 1):
        for j in range(i + 1, a.shape[0]):
            dis.append(get_distance(a[i, :], a[j, :]))

    dis = get_pairwise_distances(a)
    A=dis
    #A = np.zeros((474,474))

    #A = np.corrcoef(a)
    Z, E, ranks = __solve_low_rank_sparse_graph_LRR_representation(X, A, lamb)


    return X @ Z, E, ranks,Z

def clustering_coef_wu(W):
    W2=np.array(W)
    W2[W2==0]=0
    W2[W2!=0]=1
    K=np.sum(W2,axis=1)  #横向量之积

    w1 = np.power(W, 1/3)  #数组次方
    w1[~np.isfinite(w1)] = 0
    cyc3=np.diag(np.dot(np.dot(w1,w1),w1))  #求对角线
    K[cyc3==0]=inf
    C=cyc3/(K*(K-1))
    C[~np.isfinite(C)]=0
    return C

def cal_pcc(data, phi):
    '''
    :param data:  图   871 * 116 * ?
    :return:  adj
    '''
    corr_matrix = []
    # for key in range(len(data)):  # 每一个sample
    #     corr_mat = np.corrcoef(data[key])
    #     # if key == 5:
    #     #    print(corr_mat)
    #     corr_mat = np.arctanh(corr_mat - np.eye(corr_mat.shape[0]))
    #     corr_matrix.append(corr_mat)
    data_array = np.array(data)  # 871 116 116
    data_array[np.isnan(data_array)] = 0
    data_array[np.isinf(data_array)] = 1
    data_array[data_array > phi] = 1
    data_array[data_array < phi*(-1)] = -1
    data_array[(data_array <= phi) & (data_array >= phi*(-1))] = 0

    '''
    where_are_nan = np.isnan(data_array)  # 找出数据中为nan的
    where_are_inf = np.isinf(data_array)  # 找出数据中为inf的
    for bb in range(0, len(data)):
        for i in range(0, dim):
            for j in range(0, dim):
                if where_are_nan[bb][i][j]:
                    data_array[bb][i][j] = 0
                if where_are_inf[bb][i][j]:
                    data_array[bb][i][j] = 1
                if data_array[bb][i][j] > phi:
                    data_array[bb][i][j] = 1
                elif data_array[bb][i][j] < phi*(-1):
                    data_array[bb][i][j] = -1
                else:
                    data_array[bb][i][j] = 0
    '''
    # print(data_array[0])
    # corr_p = np.maximum(data_array, 0)  # pylint: disable=E1101
    # corr_n = 0 - np.minimum(data_array, 0)  # pylint: disable=E1101
    # data_array = [corr_p, corr_n]
    # data_array = np.array(data_array)  # 2 871 116 116
    # data_array = np.transpose(data_array, (1, 0, 2, 3))
    # data_array2 = corr_p + corr_n
    return data_array

def cal_pcc_limt(data, phi):
    '''
    :param data:  图   871 * 116 * ?
    :return:  adj
    '''

    data_array = np.array(data)  # 871 116 116
    data_array[np.isnan(data_array)] = 0
    data_array[np.isinf(data_array)] = 1
    for i in range(len(data_array)):
        x = np.sort(np.abs(data_array[i]))
        m = len(x)
        cutoff = x[int(phi * m)]
        idxes = np.where(data_array[i] <= cutoff)
        data_array[i][idxes] = 0.

    return np.array(data_array)

def test():
    # kids = ['cc200','aal','ho']
    # for kid in kids:
    #     data_dict = get_original_data_with_site_name(kid)
    #
    #
    #     datas = []
    #     sids = []
    #     lids = []
    #     labels = []
    #     length = 2
    #     lowrank = True
    #
    #
    #
    #     for site_name, site_data in data_dict.items():
    #         for data in site_data:
    #                 sids.append(data.sid)
    #                 lids.append(data.lid)
    #                 labels.append(data.label)
    #                 datas.append(data.vec_features)
    #
    #
    #     datas = np.stack(datas).T # n x m

    mat_data = scipy.io.loadmat('abide_lr_20.00.mat')
    print(mat_data['data'].shape)
        for i in range(1):
            params = [0.1]
            # params = [0.1,0.3,0.5]
            # params = [0.2, 0.4, 0.6]
            for lamb in params:
                #XZ, E, ranks, ZZ = solve_low_rank_sparse_graph_LRR_representation(datas, 0.01)
                XZ, E, ranks, ZZ = solve_low_rank_representation(datas, 0.01)
                #XZ, E, ZZ = latent_lrr(datas, lamb)
                savemat(
                    "../ABIDE/LR/norm_%s/abide_lr_%.2f.mat" % (kid, 0.01),
                    {
                        'lamb' : lamb,
                        'sids' : sids,
                        'lids' : lids,
                        'redata' : datas,
                        'data' : XZ,
                        'Z' : ZZ,
                        'E': E
                    }
                )


def test2():
    kids = ['cc200','aal','ho']
    for kid in kids:
        data_dict = get_original_data_with_site_name(kid)


        datas = []
        sids = []
        lids = []
        labels = []
        length = 2
        lowrank = True



        for site_name, site_data in data_dict.items():
            for data in site_data:
                    sids.append(data.sid)
                    lids.append(data.lid)
                    labels.append(data.label)
                    datas.append(data.vec_features)


        datas = np.stack(datas).T # n x m

        savemat(
            "../ABIDE/Orign/norm_%s/abide.mat" % (kid),
            {
                'sids' : sids,
                'lids' : lids,
                'redata' : datas
            }
        )

if __name__ == "__main__":
    test()
