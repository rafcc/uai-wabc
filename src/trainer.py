import copy
import logging.config
import os
import pickle
# for Logging handling
import sys
import time

import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import minimize

import model

logger = logging.getLogger(__name__)


def nonzero_indices(a):
    """Get an index with non-zero element.

    Parameters
    ----------
    a : numpy.ndarray
        array
    
    Returns
    -------
    np.nonzero() : numpy.ndarray
        Index with non-zero element
    """
    return (np.nonzero(a)[0])


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


def calc_diff(C_pre, C_pos, t_pre, t_pos, rss_pre, rss_pos):
    """calculate difference

    Parameters
    ----------
    C_pre : numpy.ndarray
        initialize control points
    C_pos : numpy.ndarray
        control points
    t_pre : numpy.ndarray
        initialize parameters
    t_pos : numpy.ndarray
        parameters
    rss_pre : int
        initialize rss
    rss_pos : int
        rss

    Returns
    -------
    np.abs() : numpy.ndarray
        absolute value
    """
    if t_pre.shape[1] > t_pos.shape[1]:
        t_pos = np.c_[t_pos, 1 - np.sum(t_pos, axis=1)]
    else:
        t_pre = np.c_[t_pre, 1 - np.sum(t_pre, axis=1)]
        t_pos = np.c_[t_pos, 1 - np.sum(t_pos, axis=1)]
    ratio_sum = 0
    for key in C_pre:
        ratio_sum += np.linalg.norm(C_pre[key] - C_pos[key]) / np.linalg.norm(
            C_pre[key])
    diff = rss_pre - rss_pos
    logger.debug("{} {} {}".format(rss_pre, rss_pos, diff))
    return (np.abs(diff))


def calc_gd_igd(dd1, dd2):
    """Calculate gd and igd.

    Parameters
    ----------
    dd1 : numpy.ndarray
        estimated bezier simplex sample
    dd2 : numpy.ndarray
        validation data

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


class BorgesPastvaTrainer:
    """Polynomial Regression Trainer.

    Attributes
    ----------
    dimSpace : int
        degree
    dimSimplex : int
        dimension
    degree : int
        dimension of constol point
    """
    def __init__(self, dimSpace, degree, dimSimplex):
        """Borges Pastva Trainer initialize.

        Parameters
        ----------
        dimSpace : int
            degree
        degree : int
            dimension of constol point
        dimSimplex : int
            dimension
        
        Returns
        ----------
        None
        """
        self.dimSpace = dimSpace  # degree of bezier simplex
        self.dimSimplex = dimSimplex  # dimension of bezier simplex
        self.degree = degree  # dimension of constol point
        self.bezier_simplex = model.BezierSimplex(dimSpace=self.dimSpace,
                                                  dimSimplex=self.dimSimplex,
                                                  degree=self.degree)

    def initialize_control_point(self, data):
        """Initialize control point.

        Parameters
        ----------
        data : list
            test data
        
        Returns
        ----------
        C : dict
            control point
        """
        bezier_simplex = model.BezierSimplex(dimSpace=self.dimSpace,
                                             dimSimplex=self.dimSimplex,
                                             degree=self.degree)
        C = bezier_simplex.initialize_control_point(data)
        return (C)

    def gradient(self, c, t):
        """Calculate gradient.

        Parameters
        ----------
        c : dict
            control point
        t : [t[0], t[1], t[2], t[3]]
            parameter
        
        Returns
        ----------
        g : float
            gradient
        """
        g = {}
        x = {}
        for d in range(self.dimSimplex - 1):
            x[d] = np.zeros(self.dimSpace)
        for d in range(self.dimSimplex - 1):
            for key in self.bezier_simplex.Mf_all.keys():
                for i in range(self.dimSpace):
                    x[d][i] += self.bezier_simplex.monomial_diff(
                        multi_index=key, d0=d, d1=None)(
                            *t[0:self.dimSimplex - 1]) * c[key][i]
        for d in x:
            g[(d, )] = x[d]
        return (g)

    def hessian(self, c, t):
        """Calculate hessian.

        Parameters
        ----------
        c : dict
            control point
        t : [t[0], t[1], t[2], t[3]]
            parameter
        
        Returns
        ----------
        h : dict
            hessian matrix
        """
        h = {}
        x = {}
        for d1 in range(self.dimSimplex - 1):
            for d2 in range(self.dimSimplex - 1):
                x[(d1, d2)] = np.zeros(self.dimSpace)

        for d1 in range(self.dimSimplex - 1):
            for d2 in range(self.dimSimplex - 1):
                for key in self.bezier_simplex.Mf_all.keys():
                    for i in range(self.dimSpace):
                        x[(d1, d2)][i] += self.bezier_simplex.monomial_diff(
                            multi_index=key, d0=d1, d1=d2)(
                                *t[0:self.dimSimplex - 1]) * c[key][i]
        for (d1, d2) in x:
            h[(d1, d2)] = x[(d1, d2)]
        return (h)

    def initialize_parameter(self, c, data):
        """Initialize parameter.

        Parameters
        ----------
        c : dict
            control point
        data : numpy.ndarray
            sample points
        
        Returns
        ----------
        tt_ : numpy.ndarray
            nearest parameter of each sample points
        xx_ : numpy.ndarray
            nearest points on the current bezier simplex
        """
        tt, xx = self.bezier_simplex.meshgrid(c)
        tt_ = np.empty([data.shape[0], self.dimSimplex])
        xx_ = np.empty([data.shape[0], self.dimSpace])
        for i in range(data.shape[0]):
            a = data[i, :]
            tmp = xx - a
            norm = np.linalg.norm(tmp, axis=1)
            amin = np.argmin(norm)
            tt_[i, :] = tt[amin, :]
            xx_[i, :] = xx[amin, :]
        return (tt_, xx_)

    def inner_product(self, c, t, x):
        """Inner product.

        Parameters
        ----------
        c : dict
            control point
        t : [t[0], t[1], t[2], t[3]]
            parameter
        x : numpy.ndarray
            point
        
        Returns
        ----------
        f : numpy.ndarray
            point
        """
        g = self.gradient(c, t)
        b = self.bezier_simplex.sampling(c, t)
        f = np.array(np.zeros(self.dimSimplex - 1))
        for d in range(self.dimSimplex - 1):
            f[d] = sum(g[(d, )][i] * (b[i] - x[i])
                       for i in range(self.dimSpace))
        return (f)

    def inner_product_jaccobian(self, c, t, x):
        """Inner product(jaccobian).

        Parameters
        ----------
        c : dict
            control point
        t : [t[0], t[1], t[2], t[3]]
            parameter
        x : numpy.ndarray
            point
        
        Returns
        ----------
        j : numpy.ndarray
            jaccobian matrix
        """
        g = self.gradient(c, t)
        b = self.bezier_simplex.sampling(c, t)
        h = self.hessian(c, t)
        j = np.zeros([self.dimSimplex - 1, self.dimSimplex - 1])
        for d1 in range(self.dimSimplex - 1):
            for d2 in range(self.dimSimplex - 1):
                j[d1, d2] = sum(h[(d1, d2)][i] * (b[i] - x[i]) +
                                g[(d1, )][i] * g[(d2, )][i]
                                for i in range(self.dimSpace))
        return (j)

    def newton_method(self, c, t_init, x, newton_itr=20, tolerance=10**(-5)):
        """Newton method.

        Parameters
        ----------
        c : dict
            control point
        t_init : list
            parameter
        x : numpy.ndarray
            point
        newton_itr : int
            iterate value
        tolerance : int
            tolerance
        
        Returns
        ----------
        t_k : numpy.ndarray
            output point
        """
        t_k = copy.deepcopy(t_init)
        for k in range(newton_itr):
            f = self.inner_product(c, t_k, x)
            if np.linalg.norm(f) > tolerance:
                j = self.inner_product_jaccobian(c, t_k, x)
                # for Logging handling
                try:
                    d = np.linalg.solve(j, f)
                except LinAlgError as e:
                    logger.critical("{0}".format(e))
                    logger.critical("The arguments are shown below")
                    logger.critical(j)
                    logger.critical(f)
                    sys.exit()
                t_k = t_k - d
            else:
                break
        return (t_k)

    def projection_onto_simplex(self, t):
        """Projection onto simplex.

        Parameters
        ----------
        t : list
            parameter
        
        Returns
        ----------
        res : numpy.ndarray
            parameter
        """
        if np.min(t) >= 0 and np.sum(t) <= 1:
            return (t)
        else:
            tmp = np.append(t, 1 - np.sum(t))

            def l2norm(x):
                return (np.linalg.norm(x - tmp))

            cons = []
            for i in range(self.dimSimplex):
                cons = cons + [{'type': 'ineq', 'fun': lambda x: x[i]}]
            cons = cons + [{'type': 'eq', 'fun': lambda x: -np.sum(x) + 1}]
            res = minimize(l2norm, x0=tmp, constraints=cons)
            return (res.x[0:self.dimSimplex - 1])

    def update_parameter(self, c, t_mat, data):
        """Projection onto simplex.

        Parameters
        ----------
        c : dict
            control point
        t_mat : list
            parameter
        data : list
            test data
        
        Returns
        ----------
        tt_ : numpy.ndarray
            parameter
        xx_ : numpy.ndarray
            points
        """
        tt_ = np.empty([data.shape[0], self.dimSimplex - 1])
        xx_ = np.empty([data.shape[0], self.dimSpace])
        for i in range(data.shape[0]):
            x = data[i]
            t = t_mat[i][0:self.dimSimplex - 1]
            t_hat = self.newton_method(c, t, x)
            t_hat2 = self.projection_onto_simplex(t_hat)
            x_hat = self.bezier_simplex.sampling(c, t_hat2)
            tt_[i] = t_hat2
            xx_[i] = x_hat
        return (tt_, xx_)

    def normal_equation(self, t_mat, data, c, indices_all, indices_fix):
        """Normal equation.

        Parameters
        ----------
        t_mat : list
            parameter
        data : list
            test data
        c : dict
            control point
        indices_all : list
            all index
        indices_fix : list
            fix index
        
        Returns
        ----------
        mat_l : numpy.ndarray
            output points
        mat_r : numpy.ndarray
            output points
        """
        mat_r = np.empty([t_mat.shape[0], len(indices_all) - len(indices_fix)])
        mat_l = copy.deepcopy(data)
        for i in range(t_mat.shape[0]):
            jj = 0
            for j in range(len(indices_all)):
                key = indices_all[j]
                if key not in indices_fix:
                    mat_r[i, jj] = self.bezier_simplex.monomial_diff(
                        multi_index=key, d0=None,
                        d1=None)(*t_mat[i, 0:self.dimSimplex - 1])
                    jj += 1
                if key in indices_fix:
                    mat_l[i, :] = mat_l[i] - self.bezier_simplex.monomial_diff(
                        multi_index=key, d0=None, d1=None)(
                            *t_mat[i, 0:self.dimSimplex - 1]) * c[key]
        return (mat_l, mat_r)

    def update_control_point(self, t_mat, data, c, indices_all, indices_fix):
        """Normal equation.

        Parameters
        ----------
        t_mat : list
            parameter
        data : list
            test data
        c : dict
            control point
        indices_all : list
            all index
        indices_fix : list
            fix index(control point)
        
        Returns
        ----------
        dic_c : numpy.ndarray
            output points
        """
        dic_c = {}
        for key in indices_all:
            dic_c[key] = np.empty(self.dimSpace)
        mat_l, mat_r = self.normal_equation(t_mat, data, c, indices_all,
                                            indices_fix)
        for i in range(data.shape[1]):
            y = mat_l[:, i]
            # for Logging handling
            try:
                c_hat = np.linalg.solve(np.dot(mat_r.T, mat_r),
                                        np.dot(mat_r.T, y))
            except LinAlgError as e:
                logger.critical("{0}".format(e))
                logger.critical("The arguments are shown below")
                logger.critical(np.dot(mat_r.T, mat_r))
                logger.critical(np.dot(mat_r.T, y))
                sys.exit()
            jj = 0
            for j in range(len(indices_all)):
                key = indices_all[j]
                if key in indices_fix:
                    dic_c[key][i] = c[key][i]
                if key not in indices_fix:
                    dic_c[key][i] = c_hat[jj]
                    jj += 1
        return (dic_c)

    def train(self,
              data,
              result_dir='',
              flag_write_meshgrid=1,
              C_init=None,
              indices_fix=None,
              max_iteration=30,
              tolerance=10**(-4),
              data_val=None):
        """Borges Pastva Training.

        Parameters
        ----------
        data : list
            test data
        result_dir : str(file path)
            directory name
        flag_write_meshgrid : int
            fragment
        C_init : dict
            control point
        indices_fix : list
            fix index
        max_iteration : int
            max iteration
        tolerance : int
            tolerance
        data_val
            all data

        Returns
        ----------
        C_pos : numpy.ndarray
            output points
        """

        create_directory(result_dir)
        create_directory(result_dir + '/control_points')
        create_directory(result_dir + '/meshgrid')
        start = time.time()

        # concat data
        if isinstance(data, dict):
            logger.debug("input data is dictionary!!!")
            index = 0
            for key in data:
                if len(key) == 1:
                    data[key] = data[key].reshape((1, self.dimSpace))
                if index == 0:
                    data_array = data[key]
                else:
                    data_array = np.r_[data_array, data[key]]
                index = index + 1
            data = data_array
        else:
            logger.debug("input data is ndarray!!!")
        logger.debug("datashape{}".format(data.shape))
        # initialize parameter
        C_pre = copy.deepcopy(C_init)
        tt_init, xx_pre = self.initialize_parameter(c=C_pre, data=data)
        tt_pre = tt_init
        rss_pre = 100000
        for itr in range(max_iteration):
            self.bezier_simplex.write_control_point(
                C=C_pre,
                filename=result_dir + '/control_points/control_point_itr_' +
                '{0:03d}'.format(itr))
            if flag_write_meshgrid == 1:
                self.bezier_simplex.write_meshgrid(C=C_pre,
                                                   filename=result_dir +
                                                   '/meshgrid/meshgrid_itr_' +
                                                   '{0:03d}'.format(itr))
            # update t
            tt_pos, xx_pos = self.update_parameter(c=C_pre,
                                                   t_mat=tt_pre,
                                                   data=data)
            # update control points
            C_pos = self.update_control_point(t_mat=tt_pos,
                                              data=data,
                                              c=C_pre,
                                              indices_all=list(C_pre.keys()),
                                              indices_fix=indices_fix)

            # calc rss
            rss_pos = np.linalg.norm(data - xx_pos) / data.shape[0]

            # check terminate condition
            epsilon = calc_diff(C_pre=C_pre,
                                C_pos=C_pos,
                                t_pre=tt_pre,
                                t_pos=tt_pos,
                                rss_pre=rss_pre,
                                rss_pos=rss_pos)
            if epsilon < tolerance:
                logger.debug("terminate condition was satisfied!")
                break

            tt_pre = tt_pos
            rss_pre = rss_pos
            C_pre = copy.deepcopy(C_pos)

        self.bezier_simplex.write_control_point(
            C_pos, result_dir + '/control_points/control_points_itr_' +
            '{0:03d}'.format(itr + 1))
        self.bezier_simplex.write_control_point(
            C_pos,
            result_dir + '/control_points_itr' + '{0:03d}'.format(itr + 1))

        # output time
        elapsed_time = time.time() - start
        with open(result_dir + '/time.txt', 'w') as wf:
            wf.write(str(elapsed_time) + '\n')

        xx = self.bezier_simplex.write_meshgrid(C=C_pos,
                                                filename=result_dir +
                                                '/meshgrid/meshgrid_itr_' +
                                                '{0:03d}'.format(itr + 1))
        np.savetxt(result_dir + '/meshgrid_itr' + '{0:03d}'.format(itr + 1),
                   xx)
        # calc gd and igd
        if data_val is None:
            pass
        else:
            gd, igd = calc_gd_igd(dd1=data_val, dd2=xx)
            np.savetxt(result_dir + '/gd.txt', [gd])
            np.savetxt(result_dir + '/igd.txt', [igd])
        return (C_pos)

    def write_parameter():
        pass


class InuductiveSkeltonTrainer:
    """Polynomial Regression Trainer.

    Attributes
    ----------
    dimSpace : int
        degree
    dimSimplex : int
        dimension
    degree : int
        dimension of constol point
    """
    def __init__(self, dimSpace, degree, dimSimplex):
        """Borges Pastva Trainer initialize.

        Parameters
        ----------
        dimSpace : int
            degree
        degree : int
            dimension of constol point
        dimSimplex : int
            dimension
        
        Returns
        ----------
        None
        """
        self.dimSpace = dimSpace
        self.dimSimplex = dimSimplex
        self.degree = degree
        self.bezier_simplex = model.BezierSimplex(dimSpace=self.dimSpace,
                                                  dimSimplex=self.dimSimplex,
                                                  degree=self.degree)

    def initialize_control_point(self, data):
        """Initialize control point.

        Parameters
        ----------
        data : list
            test data
        
        Returns
        ----------
        C : dict
            control point
        """
        bezier_simplex = model.BezierSimplex(dimSpace=self.dimSpace,
                                             dimSimplex=self.dimSimplex,
                                             degree=self.degree)
        C = bezier_simplex.initialize_control_point(data)
        return (C)

    def extract_corresponding_multiple_index(self, C, index_list):
        """Extract corresponding multiple index.

        Parameters
        ----------
        C : dict
            control point
        index_list : list
            index list data
        
        Returns
        ----------
        d : list
            corresponding multiple indices as set
        """
        d = set()
        for key in C:
            if set(np.array(nonzero_indices(key) + 1)).issubset(
                    set(index_list)):
                d.add(key)
        return (d)

    # Rename(
    # extract_corresponding_multiple_index_\
    #   from__set
    # -> ex_cor_mlti_id_f_set
    def ex_cor_mlti_id_f_set(self, s, index_list):
        """Extract corresponding multiple index.

        Parameters
        ----------
        s : list
            index set
        index_list : list
            index list data
        
        Returns
        ----------
        d : list
            corresponding multiple indices as set
        """
        d = set()
        for key in s:
            if set(np.array(nonzero_indices(key) + 1)).issubset(
                    set(index_list)):
                key2 = tuple(
                    np.array(key)[np.ndarray.tolist(np.array(index_list) - 1)])
                d.add(key2)
        return (d)

    # Rename(
    # extract_subproblem_control_point_from_whole_control_point
    # -> ex_sub_ctrl_pf_w_ctrl_p
    def ex_sub_ctrl_pf_w_ctrl_p(
            self,
            C,
            index_list):
        """Extract subproblem control point.

        Parameters
        ----------
        C : dict
            control point
        index_list : list
            index list data
        
        Returns
        ----------
        C_sub : dict
            output control point
        """
        C_sub = {}
        for key in C:
            if set(np.array(nonzero_indices(key) + 1)).issubset(
                    set(index_list)):
                key2 = tuple(
                    np.array(key)[np.ndarray.tolist(np.array(index_list) - 1)])
                C_sub[key2] = C[key]
        return (C_sub)

    # Rename(
    # insert_subproblem_control_point_to_whole_contol_point
    # -> in_sub_ctrl_p2_w_ctrl_p
    def in_sub_ctrl_p2_w_ctrl_p(
            self,
            C_whole, C_sub, index_list):
        """Insert subproblem control point.

        Parameters
        ----------
        C_whole : dict
            control point whole
        C_sub : dict
            control point sub
        index_list : list
            index list data
        
        Returns
        ----------
        C_whole : dict
            output control point whole
        """
        for key in C_sub:
            key_ = [0 for i in range(self.dimSimplex)]
            # print(key)
            for k in range(len(key)):
                key_[index_list[k] - 1] = key[k]
            key_ = tuple(key_)
            C_whole[tuple(key_)] = C_sub[key]
        return (C_whole)

    def train(self,
              data,
              data_val=None,
              max_iteration=30,
              tolerance=10**(-3),
              result_dir='',
              flag_write_meshgrid=0):
        """Polynomial Regression Training.

        Parameters
        ----------
        data : list
            test data
        data_val
            all data
        max_iteration : int
            max iteration
        tolerance : int
            tolerance
        result_dir : str(file path)
            directory name
        flag_write_meshgrid : int
            fragment

        Returns
        ----------
        None
        """
        create_directory(result_dir)
        create_directory(result_dir + '/whole/control_points')
        create_directory(result_dir + '/whole/meshgrid')

        C = self.initialize_control_point(data)
        freeze_multiple_index_set = set()
        loop = 0
        start = time.time()
        for dim in range(1, self.dimSpace + 1):
            for index in data:
                if len(index) == dim:
                    self.write_control_point(
                        C=C,
                        filename=result_dir +
                        '/whole/control_points/control_points_itr_' +
                        '{0:03d}'.format(loop))
                    # tt, xx = self.bezier_simplex.meshgrid(C=C)
                    if flag_write_meshgrid == 1:
                        self.bezier_simplex.write_meshgrid(
                            C=C,
                            filename=result_dir +
                            '/whole/meshgrid/meshgrid_itr_' +
                            '{0:03d}'.format(loop))
                    if len(freeze_multiple_index_set) == len(C.keys()):
                        logger.debug("finished")
                        break
                    else:
                        logger.debug("subproblem{}" .format(index))
                        target_multiple_index_set =\
                            self.extract_corresponding_multiple_index(
                                C=C,
                                index_list=index)
                        if dim >= 2:
                            subproblem_borges_pastva_trainer =\
                                BorgesPastvaTrainer(dimSpace=self.dimSpace,
                                                    dimSimplex=dim,
                                                    degree=self.degree)
                            # Rename(
                            # extract_subproblem_control_point_\
                            #   from_whole_control_point
                            # -> ex_sub_ctrl_pf_w_ctrl_p
                            C_sub = self.ex_sub_ctrl_pf_w_ctrl_p(
                                C=C,
                                index_list=index)
                            # Rename(
                            # extract_corresponding_multiple_index_\
                            #   from__set
                            # -> ex_cor_mlti_id_f_set
                            freeze_ = self.ex_cor_mlti_id_f_set(
                                s=freeze_multiple_index_set,
                                index_list=index)
                            logger.debug("freeze{}" .format(freeze_))
                            C_sub = subproblem_borges_pastva_trainer.train(
                                data=data[index],
                                C_init=C_sub,
                                max_iteration=max_iteration,
                                tolerance=tolerance,
                                indices_fix=list(freeze_),
                                result_dir=result_dir+'/subproblem_' +
                                '_'.join([str(i) for i in index]),
                                flag_write_meshgrid=flag_write_meshgrid)
                            # Rename(
                            # insert_subproblem_control_point_to_\
                            #   whole_contol_point
                            # -> in_sub_ctrl_p2_w_ctrl_p
                            C = self.in_sub_ctrl_p2_w_ctrl_p(
                                C_whole=C,
                                C_sub=C_sub,
                                index_list=index)
                        freeze_multiple_index_set =\
                            freeze_multiple_index_set.union(
                                target_multiple_index_set)
                        logger.debug("{} {} {} {}" .format(dim,
                                     index,
                                     len(freeze_multiple_index_set),
                                     len(C.keys())))
                        loop += 1
        self.write_control_point(C=C,
                                 filename=result_dir +
                                 '/whole/control_points/control_points_itr_' +
                                 '{0:03d}'.format(loop))
        self.write_control_point(C=C,
                                 filename=result_dir + '/control_points_itr_' +
                                 '{0:03d}'.format(loop))

        # output time
        elapsed_time = time.time() - start
        with open(result_dir + '/time.txt', 'w') as wf:
            wf.write(str(elapsed_time) + '\n')

        # output gd igd
        xx = self.bezier_simplex.write_meshgrid(
            C=C,
            filename=result_dir + '/whole/meshgrid/meshgrid_itr_' +
            '{0:03d}'.format(loop))
        np.savetxt(result_dir + '/meshgrid_itr_' + '{0:03d}'.format(loop), xx)
        # calc gd and igd
        if data_val is None:
            pass
        else:
            gd, igd = calc_gd_igd(dd1=data_val, dd2=xx)
            np.savetxt(result_dir + '/gd.txt', [gd])
            np.savetxt(result_dir + '/igd.txt', [igd])

    def write_control_point(self, C, filename):
        """Output control point

        Parameters
        ----------
        C : dict
            control point
        filename : str(file path and name)
            write data file name

        Returns
        ----------
        None
        """
        with open(filename, 'wb') as f:
            pickle.dump(C, f)

    def write_parameter():
        pass


if __name__ == '__main__':
    from itertools import combinations

    DEGREE = 3  # ベジエ単体の次数
    DIM_SIMPLEX = 5  # ベジエ単体の次元
    DIM_SPACE = 5  # 制御点が含まれるユークリッド空間の次元
    NG = 21
    NEWTON_ITR = 20
    MAX_ITR = 30  # 制御点の更新回数の上界

    # input data
    objective_function_indices_list = [i + 1 for i in range(DIM_SIMPLEX)]
    subproblem_indices_list = []
    for i in range(1, len(objective_function_indices_list) + 2):
        for c in combinations(objective_function_indices_list, i):
            subproblem_indices_list.append(c)
    data = {}
    for e in subproblem_indices_list:
        string = '_'.join([str(i) for i in e])
        data[e] = np.loadtxt('../data/normalized_pf/normalized_5-MED.pf_' +
                             string)

    logger.debug(data[(1, )])
    logger.debug(data[(2, )])
    logger.debug(data[(3, )])
    """
    inductive_skelton_trainer = InuductiveSkeltonTrainer(dimSpace=DIM_SPACE,
                                                        dimSimplex=DIM_SIMPLEX,
                                                        degree=DEGREE)
    C_init = inductive_skelton_trainer.initialize_control_point(data)
    print(C_init)
    inductive_skelton_trainer.extract_corresponding_multiple_index(
        C=C_init,index_list=(1,2))
    C_sub = inductive_skelton_trainer.extract_subproblem_control_point_from_\
    whole_control_point(C=C_init,index_list=(1,2))
    inductive_skelton_trainer.insert_subproblem_control_point_to_\
    whole_contol_point(C_whole=C_init,
                       C_sub=C_sub,
                       index_list=(1,2))
    inductive_skelton_trainer.train(data,result_dir='5med_pos',max_iteration=10)
    """

    borges_pastva_trainer = BorgesPastvaTrainer(dimSpace=DIM_SPACE,
                                                dimSimplex=DIM_SIMPLEX,
                                                degree=DEGREE)
    C_init = borges_pastva_trainer.bezier_simplex.initialize_control_point(
        data)

    logger.debug("initialize ")
    tt, xx = borges_pastva_trainer.initialize_parameter(c=C_init,
                                                        data=data[(1, 2, 3, 4,
                                                                   5)])
    logger.debug("{} {}".format(tt.shape, xx.shape))
    logger.debug("inner product")
    f = borges_pastva_trainer.inner_product(c=C_init,
                                            t=tt[2, :],
                                            x=data[(1, 2, 3, 4, 5)][2, :])
    logger.debug(f)

    logger.debug("gradient, hessian ")
    g = borges_pastva_trainer.gradient(c=C_init, t=tt[2, :])
    h = borges_pastva_trainer.hessian(c=C_init, t=tt[2, :])
    logger.debug(g)
    logger.debug(h)
    # j = borges_pastva_trainer.grad\
    #     ient(c=C_init,t=tt[2,:],x=data[(1,2,3,4,5)][2,:])
    # print(j)

    logger.debug("jaccobian ")
    j = borges_pastva_trainer.inner_product_jaccobian(c=C_init,
                                                      t=tt[2, :],
                                                      x=data[(1, 2, 3, 4,
                                                              5)][2, :])
    logger.debug(j)

    logger.debug("update parameter")
    tt_, xx_ = borges_pastva_trainer.update_parameter(c=C_init,
                                                      t_mat=tt,
                                                      data=data[(1, 2, 3, 4,
                                                                 5)])
    logger.debug("{} {}".format(tt_.shape, tt.shape))
    logger.debug(np.linalg.norm(xx_ - xx))
    logger.debug(np.linalg.norm(tt_[:, 0:4] - tt[:, 0:4]))

    logger.debug("update control point")
    all_index = list(C_init.keys())
    logger.debug(all_index)
    fix_index = [(3, 0, 0, 0, 0), (0, 3, 0, 0, 0), (0, 0, 3, 0, 0),
                 (0, 0, 0, 3, 0), (0, 0, 0, 0, 3)]

    C_ = borges_pastva_trainer.update_control_point(t_mat=tt_,
                                                    data=data[(1, 2, 3, 4, 5)],
                                                    c=C_init,
                                                    indices_all=all_index,
                                                    indices_fix=fix_index)
    for key in C_init:
        logger.debug("{} {}".format(key, C_init[key] - C_[key]))

    for key in C_init:
        if key not in fix_index:
            C_init[key] = C_init[key] + 0.1
    logger.debug("training")
    for key in C_init:
        logger.debug("{} {}".format(key, C_init[key]))
    C_k = borges_pastva_trainer.train(data=data[(1, 2, 3, 4, 5)],
                                      C_init=C_init,
                                      max_iteration=30,
                                      tolerance=10**(-4),
                                      indices_fix=fix_index,
                                      result_dir='../test')
    for key in C_init:
        logger.debug("{} {}".format(key, C_init[key] - C_k[key]))
