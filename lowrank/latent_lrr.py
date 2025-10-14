import numpy as np


def solve_low_rank_sparse_graph_LRR_representation(X, lamb):

    Z, L, E = latent_lrr(X,  lamb)
    return X @ Z, E ,Z


def latent_lrr(X, lamb):
# Latent Low-Rank Representation for Subspace Segmentation and Feature Extraction
# Guangcan Liu, Shuicheng Yan.ICCV
# 2011.
# Problem:
# min_Z, L, E | | Z | | _ * + | | L | | _ * +¡¡lamb | | E | | _1,
# s.t.X = XZ + LX +E.
# Solning problem by Inexact ALM

    A = X
    tol = 1e-6
    rho = 1.1
    max_mu = 1e6
    mu = 1e-6
    maxIter = 1e6
    d,n = X.shape
    d,m = A.shape
    atx = np.dot(X.T,X)
    inv_a = np.linalg.inv(A.T @ A+ np.eye(m))
    inv_b = np.linalg.inv(A @ A.T + np.eye(d))
    # Initializing optimization variables
    J = np.zeros(shape=[m, n])
    Z = np.zeros(shape=[m, n])
    L = np.zeros(shape=[d, d])
    S = np.zeros(shape=[d, d])

    # E = sparse(d, n);
    E = np.zeros(shape=[d, n])

    Y1 = np.zeros(shape=[d, n])
    Y2 = np.zeros(shape=[m, n])
    Y3 = np.zeros(shape=[d, d])

    # Start main loop
    iter = 0
    print('initial')

    while iter < maxIter:
        iter = iter + 1;
        # disp(['iter====>>>>' num2str(iter)]);
        # updating J by the Singular ValueThres holding(SVT)  operator
        temp_J = Z + np.divide(Y2,mu)
        U_J, sigma_J, V_J = np.linalg.svd(temp_J, 'econ')
        sigma_J = np.diag(sigma_J)
        svp_J = len(np.argwhere(sigma_J > 1 / mu))
        if svp_J >= 1:
            sigma_J = sigma_J[:svp_J]-1 / mu
        else:
            svp_J = 1
            sigma_J =  np.zeros((1,1))
        J = U_J[:, : svp_J] @ np.diag(np.diag(sigma_J)) @ V_J[:, : svp_J].T

        # updating S by the Singular Value Thresholding(SVT) operator
        temp_S = L + Y3 / mu
        U_S, sigma_S, V_S = np.linalg.svd(temp_S, 'econ')
        sigma_S = np.diag(sigma_S)
        svp_S = len(np.argwhere(sigma_S > 1 / mu))
        if svp_S >= 1 :
            sigma_S = sigma_S[:svp_S] - 1 / mu
        else :
            svp_S = 1
            sigma_S = np.zeros((1,1))
        S = U_S[:, : svp_S] @ np.diag(np.diag(sigma_S)) @ V_S[:,: svp_S].T

        # udpate Z
        Z = inv_a @ (atx - X.T@L@X - X.T @ E + J + (X.T @ Y1-Y2)/mu)

        # udpate L
        L = ((X - X @ Z - E) @ X.T+ S+(Y1 @ X.T - Y3) / mu)@inv_b

        # updateE
        xmaz = X - X @ Z - L @ X
        temp = xmaz + Y1 / mu

        #E = max(0, temp -lamb /mu)+min(0, temp +lamb /mu)
        te1 = temp - lamb / mu
        te1[te1 < 0] = 0
        te2 = temp + lamb / mu
        te2[te2 > 0] = 0
        E = te1 + te2

        leq1 = xmaz - E
        leq2 = Z - J
        leq3 = L - S
        max_l1 = np.max(np.max(np.abs(leq1)))
        max_l2 = np.max(np.max(np.abs(leq2)))
        max_l3 = np.max(np.max(np.abs(leq3)))

        stopC1 = max(max_l1, max_l2)
        stopC = max(stopC1, max_l3)
        if iter % 10 == 0:
            print("Iteration %d, mu:%.6f, Rank:%d, stopC1 1 violation: %.5f, stopC 2 violation: %.5f"
                  %
                  (iter, mu, np.linalg.matrix_rank(Z,tol=1e-3*np.linalg.norm(Z, 2)), stopC1, stopC)
            )
        if stopC < tol:
            print('LRR done.')
            break
        else:
            Y1 = Y1 + np.dot(mu,leq1)
            Y2 = Y2 + np.dot(mu,leq2)
            Y3 = Y3 + np.dot(mu,leq3)
            mu = min(max_mu, np.dot(mu,rho))

    return Z, L, E