# for Logging handling
import logging.config
import os
import sys
from itertools import combinations

import data

logger = logging.getLogger(__name__)

argvs = sys.argv

ORG_DATA_DIR = '../data/raw'
POS_DATA_DIR = '../data/preprocessed'


def create_directory(dir_name):
    """Create directory.

    Parameters
    ----------
    dir_name : str(file path)
        create directory name

    Returns
    -------
    None
    """
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    else:
        pass


if __name__ == '__main__':

    DATA_NAME = argvs[1]
    DIM_SIMPLEX = int(argvs[2])

    create_directory(POS_DATA_DIR)
    
    # pareto filter
    d = data.Dataset(ORG_DATA_DIR + '/' + DATA_NAME + '.pf')
    objective_function_indices_list = [i + 1 for i in range(DIM_SIMPLEX)]
    subproblem_indices_list = []
    for i in range(1, len(objective_function_indices_list) + 2):
        for c in combinations(objective_function_indices_list, i):
            subproblem_indices_list.append(c)
    logger.debug(subproblem_indices_list)
    logger.debug(d.values)
    RAW = {}
    for e in subproblem_indices_list:
        string = '_'.join([str(i) for i in e])
        RAW = {}
        logger.debug({i - 1 for i in e})
        RAW[e] = data.weak_pareto_filter(d, {i - 1 for i in e})
        logger.debug(RAW[e].values)
        if len(e) == 1:
            if RAW[e].values.shape[0] != 1:
                RAW[e].values = RAW[e].values[0]
            RAW[e].values = RAW[e].values.reshape([1, DIM_SIMPLEX])
        logger.debug(ORG_DATA_DIR + '/' + DATA_NAME + '.pf_' + string)
        RAW[e].write(ORG_DATA_DIR + '/' + DATA_NAME + '.pf_' + string)
    
    # load data sets
    ORG = {}
    for e in subproblem_indices_list:
        string = '_'.join([str(i) for i in e])
        ORG[e] = data.Dataset('../data/raw/' + DATA_NAME + '.pf_' + string)
        if len(e) == 1:
            ORG[e].values = ORG[e].values.reshape([1, DIM_SIMPLEX])
        if len(e) == DIM_SIMPLEX:
            ORG_ALL = data.Dataset('../data/raw/' + DATA_NAME + '.pf_' + string)
    logger.debug(ORG[(1, )].values.shape)
    
    # normalize data
    cmax = ORG_ALL.values.max(axis=0)
    cmin = ORG_ALL.values.min(axis=0)
    ORG_ALL.values = (ORG_ALL.values - cmin) / (cmax - cmin)
    for e in ORG:
        ORG[e].values = (ORG[e].values - cmin) / (cmax - cmin)
    
    for e in ORG:
        logger.debug("{} {}".format(e, ORG[e].values.shape))
    for e1 in ORG:
        if len(e1) >= 2:
            # drop duplicated lines
            ORG[e1].values = ORG[e1].unique()
            # remove elements which is included by the solutions of subproblems
            for e2 in subproblem_indices_list:
                if len(e2) < len(e1):
                    ORG[e1].values = ORG[e1].difference(ORG[e2])
        # write preprocessed data
        string = '_'.join([str(i) for i in e1])
        ORG[e1].write(POS_DATA_DIR + '/' + DATA_NAME + '.pf_' + string)
    ORG_ALL.values = ORG_ALL.unique()
    ORG_ALL.write(POS_DATA_DIR + '/' + DATA_NAME + '.pf')
