import glob
import logging.config
import os
import pickle
import sys
import time
from itertools import combinations

import numpy as np

import data
import poly_trainer
import trainer
import visualize

argvs = sys.argv


DEGREE = 3
NG = 21
NEWTON_ITR = 20
MAX_ITR = 100
TOL = 10**(-3)
DATA_DIR = '../data/preprocessed'


def create_directory(dir_name):
    """create directory

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

    DATA_NAME = argvs[1]  # '3-MED'
    DIM_SIMPLEX = int(argvs[2])  # 3
    DIM_SPACE = int(argvs[3])  # 3
    N2 = int(argvs[4])  # 10
    N3 = int(argvs[5])  # 10
    SEED = int(argvs[6])  # 0

    # logger object
    logger = logging.getLogger(__name__)
    # for Logging handling
    argc = len(argvs)
    if (argc < 8):  # no setting error level
        logger.debug("Usage: set default error level")
        DATA_LVL = 'INFO'
    else:
        DATA_LVL = argvs[7]  # info
    LOG_LEVEL = ['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
    DATA_LVL = DATA_LVL.upper()
    if DATA_LVL not in LOG_LEVEL:
        logger.critical("Usage: Logging Level name setting error")
        sys.exit()
    # logger level set
    logger.setLevel(DATA_LVL)
    # console output set
    handler1 = logging.StreamHandler()
    handler1.setFormatter(
        logging.Formatter(
            "%(filename)s:%(funcName)s:%(lineno)d:%(levelname)s:%(message)s"))
    logger.addHandler(handler1)
    # file output set
    handler2 = logging.FileHandler("LOGGING_OUTPUT.log")
    handler2.setFormatter(
        logging.Formatter(
            "%(filename)s:%(funcName)s:%(lineno)d:%(levelname)s:%(message)s"))
    logger.addHandler(handler2)
    # for Logging handling
    logger.info("Exec Pattern: %s %d %d %d %d %d" %
                (DATA_NAME, DIM_SIMPLEX, DIM_SPACE, N2, N3, SEED))

    if DIM_SPACE == 2:
        RESULT_DIR = '../result/' + DATA_NAME + '/n2_' + str(N2) + '/seed_' + str(
            SEED)
    else:
        RESULT_DIR = '../result/' + DATA_NAME + '/n2_' + str(N2) + ',n3_' + str(
            N3) + '/seed_' + str(SEED)
    RESULT_DIR_INDUCTIVE = RESULT_DIR + '/inductive'
    RESULT_DIR_BORGES = RESULT_DIR + '/borges'
    RESULT_DIR_POLY = RESULT_DIR + '/poly'

    create_directory(dir_name=RESULT_DIR)
    
    objective_function_indices_list = [i + 1 for i in range(DIM_SIMPLEX)]
    subproblem_indices_list = []
    for i in range(1, len(objective_function_indices_list) + 2):
        for c in combinations(objective_function_indices_list, i):
            subproblem_indices_list.append(c)
    
    start = time.time()
    
    # create train data and validation data
    DATA_ALL = data.Dataset(DATA_DIR + '/' + DATA_NAME + '.pf').values
    DATA_TRN = {}
    np.random.seed(SEED)
    for e in subproblem_indices_list:
        if len(e) <= 3:
            string = '_'.join([str(i) for i in e])
            tmp = data.Dataset(DATA_DIR + '/' + DATA_NAME + '.pf_' + string)
            if len(e) == 2:
                DATA_TRN[e] = tmp.sample(N2)
            elif len(e) == 3:
                DATA_TRN[e] = tmp.sample(N3)
            else:
                DATA_TRN[e] = tmp.values
    index = 0
    for key in DATA_TRN:
        logger.debug("{} {}".format(key, DATA_TRN[key].shape))
        if len(key) == 1:
            tmp = DATA_TRN[key].reshape((1, DIM_SPACE))
        else:
            tmp = DATA_TRN[key]
        if index == 0:
            DATA_TRN_ARRAY = tmp
        else:
            DATA_TRN_ARRAY = np.r_[DATA_TRN_ARRAY, tmp]
        index = index + 1
    
    np.savetxt(RESULT_DIR + '/' + 'data_trn', DATA_TRN_ARRAY)
    with open(RESULT_DIR + '/' + 'data_trn.pkl', 'wb') as f:
        pickle.dump(DATA_TRN, f)
    
    for e in DATA_TRN:
        if len(e) == 2:
            DATA_TRN[e] = np.r_[DATA_TRN[e], DATA_TRN[(e[0], )].reshape(
                (1, DIM_SPACE))]
            DATA_TRN[e] = np.r_[DATA_TRN[e], DATA_TRN[(e[1], )].reshape(
                (1, DIM_SPACE))]
        if len(e) == 3:
            DATA_TRN[e] = np.r_[DATA_TRN[e], DATA_TRN[(e[0], )].reshape(
                (1, DIM_SPACE))]
            DATA_TRN[e] = np.r_[DATA_TRN[e], DATA_TRN[(e[1], )].reshape(
                (1, DIM_SPACE))]
            DATA_TRN[e] = np.r_[DATA_TRN[e], DATA_TRN[(e[2], )].reshape(
                (1, DIM_SPACE))]
            DATA_TRN[e] = np.r_[DATA_TRN[e], DATA_TRN[(e[0], e[1])]]
            DATA_TRN[e] = np.r_[DATA_TRN[e], DATA_TRN[(e[1], e[2])]]
            DATA_TRN[e] = np.r_[DATA_TRN[e], DATA_TRN[(e[0], e[2])]]
    
    for e in DATA_TRN:
        logger.debug("{} {}".format(e, DATA_TRN[e].shape))
    logger.debug(DATA_ALL.shape)
    
    # for Logging handling
    logger.info("trainer.InuductiveSkeltonTrainer(dimSpace)   : %d" % DIM_SPACE)
    logger.info("trainer.InuductiveSkeltonTrainer(dimSimplex) : %d" % DIM_SIMPLEX)
    logger.info("trainer.InuductiveSkeltonTrainer(degree)     : %d" % DEGREE)
    
    # train inductive skelton trainer
    inductive_skelton_trainer = trainer.InuductiveSkeltonTrainer(
        dimSpace=DIM_SPACE, dimSimplex=DIM_SIMPLEX, degree=DEGREE)
    inductive_skelton_trainer.train(data=DATA_TRN,
                                    data_val=DATA_ALL,
                                    result_dir=RESULT_DIR_INDUCTIVE,
                                    tolerance=TOL,
                                    max_iteration=MAX_ITR,
                                    flag_write_meshgrid=0)
    for dirname in glob.glob(RESULT_DIR_INDUCTIVE + "/"):
        for fname in glob.glob(dirname + "meshgrid_itr*"):
            if ".png" not in fname:
                pngname = fname + '.png'
                pngname2 = fname + '_compare.png'
                logger.debug(pngname)
                visualize.plot_pairplot(fname, pngname)
                visualize.plot_estimated_pairplot(d1=DATA_ALL,
                                                  dname2=fname,
                                                  dname3=RESULT_DIR + '/' +
                                                  'data_trn',
                                                  ofname=pngname2)
    
    # for Logging handling
    logger.info("trainer.BorgesPastvaTrainer(dimSpace)   : %d" % DIM_SPACE)
    logger.info("trainer.BorgesPastvaTrainer(dimSimplex) : %d" % DIM_SIMPLEX)
    logger.info("trainer.BorgesPastvaTrainer(degree)     : %d" % DEGREE)
    
    # train borges pastva trainer
    borges_pastva_trainer = trainer.BorgesPastvaTrainer(dimSpace=DIM_SPACE,
                                                        dimSimplex=DIM_SIMPLEX,
                                                        degree=DEGREE)
    C_init = borges_pastva_trainer.bezier_simplex.initialize_control_point(
        data=DATA_TRN)
    borges_pastva_trainer.train(data=DATA_TRN,
                                data_val=DATA_ALL,
                                C_init=C_init,
                                indices_fix=[],
                                result_dir=RESULT_DIR_BORGES,
                                tolerance=TOL,
                                max_iteration=MAX_ITR,
                                flag_write_meshgrid=0)
    for dirname in glob.glob(RESULT_DIR_BORGES + "/"):
        for fname in glob.glob(dirname + "meshgrid_itr*"):
            if ".png" not in fname:
                pngname = fname + '.png'
                pngname2 = fname + '_compare.png'
                logger.debug(pngname)
                visualize.plot_pairplot(fname, pngname)
                visualize.plot_estimated_pairplot(d1=DATA_ALL,
                                                  dname2=fname,
                                                  dname3=RESULT_DIR + '/' +
                                                  'data_trn',
                                                  ofname=pngname2)
    elapsed_time = time.time() - start
    logger.info("-------------")
    logger.info(" time : %f" % elapsed_time)
    logger.info("-------------")
    
    # train polynomial regression
    if (DATA_NAME != "5-MED-graph"):
        # for Logging handling
        logger.info("poly_trainer.PolynomialRegressionTrainer(dimSpace) : %d" %
                    DIM_SPACE)
        logger.info("poly_trainer.PolynomialRegressionTrainer(ng)       : %d" % NG)
        logger.info("poly_trainer.PolynomialRegressionTrainer(flag_2d)  : %d" % 1)
    
        polynomial_regression_trainer = poly_trainer.PolynomialRegressionTrainer(
            dimSpace=DIM_SPACE, ng=NG, flag_2d=1)
        polynomial_regression_trainer.train(data_trn=DATA_TRN,
                                            data_val=DATA_ALL,
                                            target_index=-1,
                                            result_dir=RESULT_DIR_POLY)
        visualize.plot_pairplot(RESULT_DIR_POLY + "/meshgrid",
                                RESULT_DIR_POLY + "/meshgrid.png")
        visualize.plot_estimated_pairplot(d1=DATA_ALL,
                                          dname2=RESULT_DIR_POLY + "/meshgrid",
                                          dname3=RESULT_DIR + '/data_trn',
                                          ofname=RESULT_DIR_POLY +
                                          "/meshgrid_compare.png")
    
        elapsed_time = time.time() - start
        logger.info("-------------")
        logger.info(" time : %f" % elapsed_time)
        logger.info("-------------")
