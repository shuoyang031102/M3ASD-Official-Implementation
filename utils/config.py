class CONFIG:
    SITES = ['NYU', "UCLA", "USM", "UM", "Leuven"]
    #SITES = ['NYU', "UM", "Leuven", "USM", "UCLA","Yale"]
    #SITES = ["Yale", "Pitt", "SBL", "Stanford", "Caltech", "NYU", "UM", "MaxMun", "Olin", "USM", "UCLA", "Leuven", "Trinity", "KKI", "SDSU", "OHSU"]
    #SITES = ["Yale", "Pitt", "SBL", "Stanford", "Caltech", "NYU", "UCLA", "Leuven", "Trinity", "KKI", "SDSU", "OHSU"]
    CORRELATION = 'correlation'
    NORM = True

    MAX_EPOCHES = 60
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32

    CUTOFF_PERCENTILE = 0.0

class GINParams:
    num_mlp_layer = 2
    hidden_dims = [64, 16, 16, 16]
    final_dropout = 0.5
    learn_eps = True
    graph_pooling_type = 'sum'
    neighbor_pooling_type = 'sum'
    input_type = 'indicator'

class GCNParams:
    knn_k = 6
    hidden_dims = [128, 64, 32, 16]
    cheby_k = 5

class DNNParams:
    hidden_dims = [256, 32, 2]

class axis_length:
    length={'ho': 110, 'aal': 116, 'aal116': 116, 'aal90': 90, 'aal64':64,'cc200':200}
"""
Yale, 47
Pitt, 45
SBL, 26
Stanford, 36
Caltech, 37
NYU, 170
UM, 107
MaxMun, 42
Olin, 25
USM, 61
UCLA, 75
Leuven, 61
Trinity, 44
KKI, 38
SDSU, 33
OHSU, 23
CMU, 5
"""