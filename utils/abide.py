import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf
from scipy.io import loadmat


from data_utils import get_percentile_value, get_ids, get_subject_label, fetch_conn_matrices, get_atlas_coords
from config import CONFIG
from spectral import distance_scipy_spatial, adjacency, laplacian
from config import CONFIG, GCNParams, axis_length


class ABIDEGraph():
    def __init__(self, adj, label, site_name, sid, lid):
        self.sid = sid
        self.lid = lid
        self.adj = tf.cast(tf.constant(adj), dtype=tf.float32)
        self.label = label
        self.site_name = site_name

        num_nodes, _ = adj.shape
        self.indicator = tf.constant(np.eye(num_nodes), dtype=tf.float32)
        self.degrees = tf.constant(
            np.diag(
                np.sum(adj, axis=1, keepdims=False)
            ), dtype=tf.float32
        )

        idx = np.triu_indices_from(adj,1)
        self.vec_features = tf.cast(
            tf.constant(adj[idx]), dtype=tf.float32
        )


def get_original_data_with_site_name(atlas_name):
    subject_ids_short = get_ids(None, True)
    subject_ids_long = get_ids(None, False)

    idx = []
    for i, id in enumerate(subject_ids_long):
        if any(s in id for s in CONFIG.SITES):
            idx.append(i)

    subject_ids_short = subject_ids_short[idx]
    subject_ids_long = subject_ids_long[idx]
    label_dict = get_subject_label(subject_ids_short, label_name='DX_GROUP')
    adjs = fetch_conn_matrices(subject_ids_short, atlas_name=atlas_name, kind=CONFIG.CORRELATION, norm=CONFIG.NORM)

    cutoff = get_percentile_value(adjs, percentile=CONFIG.CUTOFF_PERCENTILE)
    idxes = np.where(np.abs(adjs) <= cutoff)
    adjs[idxes] = 0.

    ret = {}

    m = len(subject_ids_short)
    for i in range(m):
        site_name = subject_ids_long[i].split('_')[0]
        label = -1 * int(label_dict[subject_ids_short[i]]) + 2
        adj = adjs[i]
        data = ABIDEGraph(adj, label, site_name, subject_ids_short[i], subject_ids_long[i])
        if site_name not in ret.keys():
            ret[site_name] = [data]
        else:
            ret[site_name].append(data)
    return ret

def get_low_rank_data_with_site_name(atlas_name,lamb,dir = ''):
    mdict =[]
    if(dir==''):
        mdict = loadmat('../ABIDE/lowrank/abide_lr_%.2f.mat' % (lamb))
    else:
        mdict = loadmat(dir)
    short_ids = mdict['sids']
    long_ids = mdict['lids']
    data = mdict['data']
    length = axis_length.length[atlas_name]
    label_dict = get_subject_label(short_ids, 'DX_GROUP')

    idx = np.triu_indices(length, k=1)

    ret = {}
    for i, lid in enumerate(long_ids):
        site_name = lid.split('_')[0]
        label = -1 * int(label_dict[short_ids[i]]) + 2

        conn = np.zeros(shape=[length, length])
        conn[idx] = data[:,i]
        conn += conn.T
        conn += np.eye(length)
        abidegraph = ABIDEGraph(conn, label, site_name, short_ids[i], long_ids[i])

        if site_name not in ret.keys():
            ret[site_name] = [abidegraph]
        else:
            ret[site_name].append(abidegraph)
    return ret

def get_low_rank_data_with_site_nameBydir(atlas_name,lamb,dir = ''):
    mdict =[]
    if(dir==''):
        mdict = loadmat('../ABIDE/lowrank/abide_lr_%.2f.mat' % (lamb))
    else:
        mdict = loadmat(dir)
    short_ids = mdict['sids']
    long_ids = mdict['lids']
    data = mdict['data']
    length = axis_length.length[atlas_name]
    label_dict = get_subject_label(short_ids, 'DX_GROUP')

    idx = np.triu_indices(length, k=1)

    ret = {}
    for i, lid in enumerate(long_ids):
        site_name = lid.split('_')[0]
        label = -1 * int(label_dict[short_ids[i]]) + 2

        conn = np.zeros(shape=[length, length])
        conn[idx] = data[:,i]
        conn += conn.T
        conn += np.eye(length)
        abidegraph = ABIDEGraph(conn, label, site_name, short_ids[i], long_ids[i])

        if site_name not in ret.keys():
            ret[site_name] = [abidegraph]
        else:
            ret[site_name].append(abidegraph)
    return ret


def get_data_with_site_name(kind, atlas_name, lamb=None, dir = []):
    if kind is 'original':
        return get_original_data_with_site_name(atlas_name)
    elif kind is 'lowrank':
        length = len(dir)
        if length != 0:
            datadir =[]
            for i in range(length):
                data = get_low_rank_data_with_site_name('aal', 0.01, dir[i])
                datadir.append(data)
            ret = {}
            for data_dict in datadir:
                for site_name, site_data in data_dict.items():
                    if site_name not in ret.keys():
                        ret[site_name] = [site_data]
                    else:
                        ret[site_name].append(site_data)
            return ret
        else:
            return get_low_rank_data_with_site_name(atlas_name,lamb)

class ABIDE(object):
    def __init__(self, val_site, atlas_name, kind='original', lamb=None,dir = []):
        self.coords = get_atlas_coords(atlas_name)
        dists, idx = distance_scipy_spatial(self.coords, GCNParams.knn_k)
        self.adj = adjacency(dists, idx)
        print(self.adj.shape)
        self.val_site = val_site
        self.kind = kind

        self.data = get_data_with_site_name(kind=kind, atlas_name=atlas_name, lamb=lamb, dir=dir)

        if type(val_site) is str:
            self.val_data = self.data[val_site]
            self.train_data = []
            for key in self.data.keys():
                if key != val_site:
                    self.train_data.extend(self.data[key])
            self.num_train = len(self.train_data)
            random_idx = np.random.permutation(self.num_train)
            self.train_data = [self.train_data[i] for i in random_idx]

        else:
            m = sum([len(v) for k,v in self.data.items()])
            self.fold_size = int(m * 0.1)
            self.fold_idx = val_site

            data = []
            for k in self.data.keys():
                data.extend(self.data[k])

            self.train_data = data[0:self.fold_size * self.fold_idx] + data[self.fold_size * (self.fold_idx + 1):]
            self.val_data = data[self.fold_size * self.fold_idx: self.fold_size * (self.fold_idx + 1)]
            self.num_train = len(self.train_data)

        self.cursor = 0
        self.epoches = 0

    def load_batch(self, batchsize):
        ret = []
        for _ in range(batchsize):
            ret.append(self.train_data[self.cursor])
            self.cursor += 1
            if self.cursor == self.num_train:
                self.cursor = 0
                self.epoches += 1
        return ret, np.array([g.label for g in ret])

    def load_test(self):
        return self.val_data, np.array([g.label for g in self.val_data])

    def get_num_epoch(self):
        return self.epoches + float(self.cursor) / self.num_train

if __name__ == "__main__":
    abide = ABIDE(val_site=['NYU','UM'], atlas_name = 'aal')