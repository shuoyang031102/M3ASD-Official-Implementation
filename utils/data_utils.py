import sys
sys.path.append('..')

import os
import csv
import numpy as np
import scipy.io as sio


from sklearn.covariance import GraphicalLassoCV
import nilearn
from nilearn import connectome

import tensorflow as tf

from spectral import distance_scipy_spatial, adjacency
from config import CONFIG, GCNParams

root_folder = "../ABIDE"
# root_folder = '/kaggle/input/abide-data/ABIDE/ABIDE'

def get_ids(num_subjects=None, short=True):
    """
        num_subjects   : number of subject IDs to get
        short          : True of False, specifies whether to get short or long subject IDs

    return:
        subject_IDs    : list of subject IDs (length num_subjects)
    """

    if short:
        subject_IDs = np.loadtxt(os.path.join(root_folder, 'subject_IDs.txt'), dtype=int)
        subject_IDs = subject_IDs.astype(str)
    else:
        subject_IDs = np.loadtxt(os.path.join(root_folder, 'full_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs

def fetch_filenames(subject_list, file_type, short=False):
    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types

    returns:

        filenames    : list of filetypes (same length as subject_list)
    """

    # Specify file mappings for the possible file types
    filemapping = {'func_preproc':'_func_preproc.nii.gz',
                   'rois_aal':'_rois_aal.1D',
                   'rois_cc200':'_rois_cc200.1D',
                   'rois_ho':'_rois_ho.1D',
                   'ho_correlation':'_ho_correlation.mat',
                   'ho_partial_correlation': '_ho_partial_correlation.mat',
                   'norm_ho_correlation':'_norm_ho_correlation.mat',
                   'norm_ho_partial_correlation':'_norm_ho_partial_correlation.mat',
                   'aal_correlation': '_aal_correlation.mat',
                   'aal_partial_correlation': '_aal_partial_correlation.mat',
                   'norm_aal_correlation': '_norm_aal_correlation.mat',
                   'norm_aal_partial_correlation': '_norm_aal_partial_correlation.mat',
                   'aal90_correlation': '_aal90_correlation.mat',
                   'aal90_partial_correlation': '_aal90_partial_correlation.mat',
                   'norm_aal90_correlation': '_norm_aal90_correlation.mat',
                   'norm_aal90_partial_correlation': '_norm_aal90_partial_correlation.mat',
                   'cc200_correlation': '_cc200_correlation.mat',
                   'cc200_partial_correlation': '_cc200_partial_correlation.mat',
                   'norm_cc200_correlation': '_norm_cc200_correlation.mat',
                   'norm_cc200_partial_correlation': '_norm_cc200_partial_correlation.mat',
                   }
    filenames = []
    subject_IDs = get_ids(short=True)
    subject_IDs = subject_IDs.tolist()
    full_IDs = get_ids(short=False)

    for s in subject_list:

        if file_type in filemapping:
            idx = subject_IDs.index(s)
            if not short:
                pattern = full_IDs[idx] + filemapping[file_type]  # Yale_0050557_rois_ho.1D
            else:
                pattern = subject_IDs[idx] + filemapping[file_type]

            filenames.append(
                os.path.join(root_folder, file_type, pattern)
            )
            # print(
            #     os.path.join(root_folder, file_type, pattern)
            # )

    return filenames

def fetch_subject_files(subjectID):
    """
        subjectID : short subject ID for which list of available files are fetched

    returns:

        onlyfiles : list of absolute paths for available subject files
    """

    # Load subject ID lists
    subject_IDs = get_ids(short=True)
    subject_IDs = subject_IDs.tolist()
    full_IDs = get_ids(short=False)

    try:
        idx = subject_IDs.index(subjectID)
        subject_folder = os.path.join(root_folder, subjectID)
        onlyfiles = [os.path.join(subject_folder, f) for f in os.listdir(subject_folder)
                     if os.path.isfile(os.path.join(subject_folder, f))]
    except ValueError:
        onlyfiles = []

    return onlyfiles

def fetch_conn_matrices(subject_list, atlas_name, kind, norm=False):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200
        kind         : the kind of correlation used to estimate the matrices, i.e.

    returns:
        connectivity : list of square connectivity matrices, one for each subject in subject_list
    """
    norm = "norm_" if norm else ""
    c = norm + atlas_name + '_' + kind.replace(' ', '_')
    conn_files = fetch_filenames(subject_list,
                                 norm + atlas_name + '_' + kind.replace(' ', '_'), short=True)
    conn_matrices = []

    for fl in conn_files:
        print("Reading connectivity file %s" % fl)
        try:
            mat = sio.loadmat(fl)['connectivity']
            if atlas_name == 'ho':
                mat = np.delete(mat, 82, axis=0)
                mat = np.delete(mat, 82, axis=1)
            conn_matrices.append(mat)
        except IOError:
            print("File %s does not exist" % fl)

    return np.array(conn_matrices)

def get_timeseries(subject_list, atlas_name):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200

    returns:
        ts           : list of timeseries arrays, each of shape (timepoints x regions)
    """

    ts_files = fetch_filenames(subject_list, 'rois_' + atlas_name)

    ts = []

    for fl in ts_files:
        print("Reading timeseries file %s" % fl)
        ts.append(np.loadtxt(fl, skiprows=0))

    return ts

def norm_timeseries(ts_list):
    """
        ts_list    : list of timeseries arrays, each of shape (timepoints x regions)

    returns:
        norm_ts    : list of normalised timeseries arrays, same shape as ts_list
    """

    norm_ts = []

    for ts in ts_list:
        norm_ts.append(nilearn.signal.clean(ts, detrend=False))

    return norm_ts

def subject_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path=root_folder):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subject      : the subject short ID
        atlas_name   : name of the atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    print("Estimating %s matrix for subject %s" % (kind, subject))

    if kind == 'lasso':
        # Graph Lasso estimator
        # covariance_estimator = GraphLassoCV(verbose=1)
        covariance_estimator = GraphicalLassoCV(verbose=1)
        covariance_estimator.fit(timeseries)
        connectivity = covariance_estimator.covariance_
        print('Covariance matrix has shape {0}.'.format(connectivity.shape))

    elif kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        subject_file = os.path.join(save_path, subject,
                                    subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        print(subject_file)
        # sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity

def group_connectivity(timeseries, subject_list, atlas_name, kind, save=True, save_path=root_folder, norm=False):
    """
        timeseries   : list of timeseries tables for subjects (timepoints x regions)
        subject_list : the subject short IDs list
        atlas_name   : name of the atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    if kind == 'lasso':
        # Graph Lasso estimator
        covariance_estimator = GraphicalLassoCV(verbose=1)
        connectivity_matrices = []

        for i, ts in enumerate(timeseries):
            print(ts)
            covariance_estimator.fit(ts)
            connectivity = covariance_estimator.covariance_
            connectivity_matrices.append(connectivity)
            print('Covariance matrix has shape {0}.'.format(connectivity.shape))

    elif kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity_matrices = conn_measure.fit_transform(timeseries)

    if save:
        for i, subject in enumerate(subject_list):
            norm = "norm_" if norm else ""
            subject_file = os.path.join(save_path, norm + atlas_name + "_" + kind.replace(' ', '_'),
                                        subject_list[i] + '_' + norm + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
            sio.savemat(subject_file, {'connectivity': connectivity_matrices[i]})
            print("Saving connectivity matrix to %s" % subject_file)

    return connectivity_matrices

def get_subject_label(subject_list, label_name):
    """
        subject_list : the subject short IDs list
        label_name   : name of the label to be retrieved

    returns:
        label        : dictionary of subject labels
    """

    label = {}

    with open(os.path.join(root_folder,'Phenotypic_V1_0b_preprocessed1.csv')) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row['subject'] in subject_list:
                label[row['subject']] = row[label_name]

    return label

def load_all_networks(subject_list, kind, atlas_name="aal"):
    """
        subject_list : the subject short IDs list
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the atlas used

    returns:
        all_networks : list of connectivity matrices (regions x regions)
    """

    all_networks = []

    for subject in subject_list:
        fl = os.path.join(root_folder, atlas_name + "_" + kind,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)['connectivity']

        if atlas_name == 'ho':
            matrix = np.delete(matrix, 82, axis=0)
            matrix = np.delete(matrix, 82, axis=1)

        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    return all_networks

def get_net_vectors(subject_list, kind, atlas_name="aal"):
    """
        subject_list : the subject short IDs list
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the atlas used

    returns:
        matrix       : matrix of connectivity vectors (num_subjects x num_connections)
    """

    # This is an alternative implementation
    networks = load_all_networks(subject_list, kind, atlas_name=atlas_name)
    # Get Fisher transformed matrices
    norm_networks = [np.arctanh(mat) for mat in networks]
    # Get upper diagonal indices
    idx = np.triu_indices_from(norm_networks[0], 1)
    # Get vectorised matrices
    vec_networks = [mat[idx] for mat in norm_networks]
    # Each subject should be a row of the matrix
    matrix = np.vstack(vec_networks)

    return matrix

def get_atlas_coords(atlas_name='ho'):
    """
        atlas_name   : name of the atlas used

    returns:
        matrix       : matrix of roi 3D coordinates in MNI space (num_rois x 3)
    """

    coords_file = os.path.join(root_folder, atlas_name + '_coords.csv')
    coords = np.loadtxt(coords_file, delimiter=',')

    if atlas_name == 'ho':
        coords = np.delete(coords, 82, axis=0)



    return coords

def get_percentile_value(x:np.array, percentile):
    x = np.sort(np.abs(np.ravel(x)))
    m = len(x)
    v = x[int(percentile * m)]
    return v

class ABIDEGraph():
    def __init__(self, adj, label, site_name):
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


if __name__ == '__main__':
    short_ids = get_ids(short=True)
    long_ids = get_ids(short=False)
    site = 'Leuven'

    asd_ages = []
    nc_ages = []

    asd_male = 0
    asd_female = 0
    nc_male = 0
    nc_female = 0

    label_dict = get_subject_label(short_ids, label_name='DX_GROUP')
    age_dict = get_subject_label(short_ids, label_name='AGE_AT_SCAN')
    sex_dict = get_subject_label(short_ids, label_name='SEX')

    for i in range(len(short_ids)):
        lid = long_ids[i]
        sid = short_ids[i]

        site_name = lid.split('_')[0]

        if site_name != site:
            continue

        label = label_dict[sid]
        if label == '1':
            asd_ages.append(
                float(age_dict[sid])
            )
            if sex_dict[sid] == '1':
                asd_male += 1
            else:
                asd_female += 1
        else:
            nc_ages.append(float(age_dict[sid]))
            if sex_dict[sid] == '1':
                nc_male += 1
            else:
                nc_female += 1

    print(asd_male, asd_female, nc_male, nc_female)
    print(np.mean(asd_ages), np.std(asd_ages), np.mean(nc_ages), np.std(nc_ages))



