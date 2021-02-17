# for Logging handling
import logging.config

import numpy as np

logger = logging.getLogger(__name__)


def calc_gd_igd(dd1, dd2):
    """Calculate gd and igd.

    Parameters
    ----------
    dd1 : numpy.ndarray
        input sample
    dd2 : numpy.ndarray
        estimated bezier simplex sample

    Returns
    -------
    gd : float
        Generational Distance
    igd : float
        Inverted Generational Distance
    """
    gd = 0
    igd = 0
    for i in range(dd2.shape[0]):
        d2 = dd2[i, :]
        tmp = dd1 - d2
        norm = np.linalg.norm(tmp, 1, axis=1)
        v = np.min(norm)
        gd += v
    for i in range(dd1.shape[0]):
        d1 = dd1[i, :]
        tmp = dd2 - d1
        norm = np.linalg.norm(tmp, 1, axis=1)
        v = np.min(norm)
        igd += v
    return (gd / dd2.shape[0], igd / dd1.shape[0])


"""
d = np.loadtxt('../data/normalized_pf/normalized_5-MED.pf_1_2_3_4_5')
dn_b = '../result/estimated_bezier_simplex_bottomup/bottomup'
d_b = np.loadtxt(dn_b)
gd, igd = calc_gd_igd(d, d_b)
logger.debug('bottoup')
logger.debug(gd)
logger.debug(igd)

logger.debug('\nallatonce')
for i in range(31):
    dn_a = '../result/estimated_bezier_simplex_allatonce'\
            '/normalized_5-MED.pf_1_2_3_4_5_itr' + str(i)
    d_a = np.loadtxt(dn_a)
    gd, igd = calc_gd_igd(d, d_a)
    logger.debug(i)
    logger.debug(gd)
    logger.debug(igd)
"""
