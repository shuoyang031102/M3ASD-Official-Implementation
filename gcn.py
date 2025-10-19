import sys


import tensorflow as tf

import scipy.sparse
import numpy as np

from mlp import DenseNormRelu
sys.path.append('../utils/')
from ..utils.spectral import laplacian, rescale_L, fourier
from ..utils.config import  GCNParams

class GraphConvolutionNet(tf.keras.Model):
    def __init__(self, adj:scipy.sparse.csr_matrix, hidden_dims):
        super(GraphConvolutionNet, self).__init__()
        L = laplacian(adj, normalized=True)
        L = rescale_L(L, lmax=2)
        lamb, U = fourier(L)
        self.L = tf.constant(L.toarray(), tf.float32)
        self.U = tf.constant(U, dtype=tf.float32)
        #print(L,U)
        self.hidden_dims = hidden_dims
        self.cheby_k = GCNParams.cheby_k
        self.dense_norm_relus = []

        for dim in self.hidden_dims[:-1]:
            self.dense_norm_relus.append(
                tf.keras.Sequential([
                    tf.keras.layers.Dense(dim),
                    tf.keras.layers.ReLU()
                ])
            )

        self.final_linear = tf.keras.layers.Dense(hidden_dims[-1])
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

    def chebyshev(self, signals):
        num_samples, num_nodes, num_features = signals.shape
        ret = []
        for i in range(num_samples):
            Xt = []
            Xt.append(signals[i])
            Xt.append(tf.matmul(self.L, signals[i]))

            for k in range(2, self.cheby_k):
                Xt.append(
                    2 * tf.matmul(self.L, Xt[k - 1]) - Xt[k - 2]
                )
            Xt = tf.stack(Xt, axis=0)
            ret.append(Xt)
        ret = tf.stack(ret, axis=0)
        ret = tf.cast(ret, dtype=tf.float32) # [num_samples, K, num_nodes, num_features]
        ret = tf.transpose(ret, perm=[0, 2, 3, 1])  # [num_samples, num_nodes, num_features, K]
        ret = tf.reshape(ret, shape=[num_samples, num_nodes, -1])
        return ret


    def call(self, inputs, training=None, mask=None):
        signals = tf.stack([g.adj for g in inputs])
        # signals = tf.constant(signals, dtype=tf.float32)
        #print(signals.shape)
        for i, drn in enumerate(self.dense_norm_relus):
            signals = self.chebyshev(signals)
            signals = drn(signals, training=training)
        batch_size = signals.shape[0]
        h = tf.reshape(signals, [batch_size, -1])
        h = self.final_linear(h)
        return self.dropout(h, training=training)
