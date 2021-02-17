# for Logging handling
import logging.config
import pickle
from functools import lru_cache

import numpy as np
import sympy

logger = logging.getLogger(__name__)


def c(ixs):
    """The sum of a range of integers

    Parameters
    ----------
    ixs : list
        data list

    Returns
    ----------
    sum : int
        values
    """
    return sum(range(1, sum((i > 0 for i in ixs)) + 1))


def BezierIndex(dim, deg):
    """Iterator indexing control points of a Bezier simplex.

    Parameters
    ----------
    dim : int
        Number of dimensions
    deg : int
        Number of degree

    Returns
    -------
    None
    """

    def iterate(c, r):
        if len(c) == dim - 1:
            yield c + (r, )
        else:
            for i in range(r, -1, -1):
                yield from iterate(c + (i, ), r - i)

    yield from iterate((), deg)


def count_nonzero(a):
    """Count the number of elements whose value is not 0.

    Parameters
    ----------
    a : numpy.ndarray
        array

    Returns
    -------
    np.count_nonzero() : int
        Count the number
    """
    return (np.count_nonzero(a))


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


def construct_simplex_meshgrid(ng, dimSimplex):
    """Construct a mesh grid.

    Parameters
    ----------
    ng : int
        Number of elements in the arithmetic progression.
    dimSimplex : int
        Number of dimensions

    Returns
    -------
    m[] : numpy.ndarray
        Array of mesh grid
    """
    t_list = np.linspace(0, 1, ng)
    tmp = np.array(np.meshgrid(*[t_list for i in range(dimSimplex - 1)]))
    m = np.zeros([tmp[0].ravel().shape[0], dimSimplex])
    for i in range(dimSimplex - 1):
        m[:, i] = tmp[i].ravel()
    m[:, dimSimplex - 1] = 1 - np.sum(m, axis=1)
    return (m[m[:, -1] >= 0, :])


class BezierSimplex:
    """BezierSimplex. access subsample data.

    Attributes
    ----------
    dimSpace : int
        degree of bezier simplex
    dimSimplex : int
        dimension of bezier simplex
    degree : int
        dimension of constol point
    """
    def __init__(self, dimSpace, dimSimplex, degree):
        """BezierSimplex initialize.

        Parameters
        ----------
        dimSpace : int
            degree of bezier simplex
        dimSimplex : int
            dimension of bezier simplex
        degree : int
            dimension of constol point

        Returns
        ----------
        None
        """
        self.dimSpace = dimSpace  # degree of bezier simplex
        self.dimSimplex = dimSimplex  # dimension of bezier simplex
        self.degree = degree  # dimension of constol point
        self.define_monomial(dimSpace=dimSpace,
                             dimSimplex=dimSimplex,
                             degree=degree)

    def define_monomial(self, dimSpace, dimSimplex, degree):
        """Define of monomial for BezierSimplex.

        Parameters
        ----------
        dimSpace : int
            degree of bezier simplex
        dimSimplex : int
            dimension of bezier simplex
        degree : int
            dimension of constol point

        Returns
        ----------
        None
        """
        T = [sympy.Symbol('t' + str(i)) for i in range(self.dimSimplex - 1)]

        def poly(i, n):
            eq = c(i)
            for k in range(n):
                eq *= (T[k]**i[k])
            return eq * (1 - sum(T[k] for k in range(n)))**i[n]

        '''M[multi_index]'''
        M = {
            i: poly(i, self.dimSimplex - 1)
            for i in BezierIndex(dim=self.dimSimplex,
                                 deg=self.degree)
        }
        '''Mf[multi_index]'''
        Mf = {}
        for i in BezierIndex(dim=self.dimSimplex, deg=self.degree):
            f = poly(i, self.dimSimplex - 1)
            b = compile('Mf[i] = lambda t0, t1=None, t2=None, t3=None: ' +
                        str(f),
                        '<string>',
                        'exec',
                        optimize=2)
            exec(b)
        '''Mf_DIFF[multi_index][t]'''
        M_DIFF = [{k: sympy.diff(v, t)
                   for k, v in M.items()} for j, t in enumerate(T)]
        Mf_DIFF = {}
        for k, v in M.items():
            Mf_DIFF[k] = []
            for j, t in enumerate(T):
                Mf_DIFF[k].append([])
                f = sympy.diff(v, t)
                b = compile(
                    'Mf_DIFF[k][-1] = lambda t0, t1=None, t2=None, t3=None: ' +
                    str(f),
                    '<string>',
                    'exec',
                    optimize=2)
                exec(b)
        '''Mf_DIFF2[multi_index][t][t]'''
        Mf_DIFF2 = {}
        for k, v in M.items():
            Mf_DIFF2[k] = []
            for h, t in enumerate(T):
                Mf_DIFF2[k].append([])
                for j in range(self.dimSimplex - 1):
                    Mf_DIFF2[k][-1].append([])
                    f = sympy.diff(M_DIFF[j][k], t)
                    b = compile(
                        'Mf_DIFF2[k][-1][-1] = '
                        'lambda t0, t1=None, t2=None, t3=None: '
                        + str(f),
                        '<string>',
                        'exec',
                        optimize=2)
                    exec(b)
        Mf_all = {}
        for k, v in M.items():
            Mf_all[k] = {}
            Mf_all[k][None] = {}
            Mf_all[k][None][None] = Mf[k]
            for i in range(len(Mf_DIFF[k])):
                Mf_all[k][i] = {}
                Mf_all[k][i][None] = Mf_DIFF[k][i]
            for i in range(len(Mf_DIFF2[k])):
                for j in range(len(Mf_DIFF2[k][i])):
                    Mf_all[k][i][j] = Mf_DIFF2[k][i][j]
        self.Mf_all = Mf_all

    @lru_cache(maxsize=1000)
    def monomial_diff(self, multi_index, d0=None, d1=None):
        """Difference of monomial for BezierSimplex.

        Parameters
        ----------
        multi_index : int
            dimention
        d0 : int
            dimention
        d1 : int
            dimention

        Returns
        ----------
        Mf_all : numpy.ndarray
            difference data of monomial
        """
        return (self.Mf_all[multi_index][d0][d1])

    def sampling(self, c, t):
        """Sampling result

        Parameters
        ----------
        c : dict
            control point
        t : [t[0], t[1], t[2], t[3]]
            parameter

        Returns
        ----------
        x : numpy.ndarray
            sampling result
        """
        x = np.zeros(self.dimSpace)
        for key in BezierIndex(dim=self.dimSimplex,
                               deg=self.degree):
            for i in range(self.dimSpace):
                x[i] += self.monomial_diff(key, d0=None, d1=None)(
                    *t[0:self.dimSimplex - 1]) * c[key][i]
        return (x)

    def sampling_array(self, c, t):
        """Sampling result

        Parameters
        ----------
        c : dict
            control point
        t : numdata*dimsimplex
            parameter
        Returns
        ----------
        x : numpy.ndarray
            sampling result
        """
        x = np.zeros((t.shape[0],self.dimSpace))
        for i in range(t.shape[0]):
            for key in BezierIndex(dim=self.dimSimplex,
                                   deg=self.degree):
                for j in range(self.dimSpace):
                    x[i,j] += self.monomial_diff(key, d0=None, d1=None)(
                        *t[i,0:self.dimSimplex - 1]) * c[key][j]
        return (x)

    def meshgrid(self, c):
        """Meshgrid

        Parameters
        ----------
        c : dict
            control point

        Returns
        ----------
        tt : numpy.ndarray
            Array of mesh grid
        xx : numpy.ndarray
            The concatenated array
        """
        tt = construct_simplex_meshgrid(21, self.dimSimplex)
        for i in range(tt.shape[0]):
            t = tt[i, :]
            if i == 0:
                x = self.sampling(c, t)
                xx = np.zeros([1, self.dimSpace])
                xx[i, :] = x
            else:
                x = self.sampling(c, t)
                x = x.reshape(1, self.dimSpace)
                xx = np.concatenate((xx, x), axis=0)
        return (tt, xx)

    def initialize_control_point(self, data):
        """Initialize control point

        Parameters
        ----------
        data : list
            test data

        Returns
        ----------
        C : dict
            control point
        """
        data_extreme_points = {}
        if isinstance(data, dict) == True:
            logger.debug(data.keys())
            for i in range(self.dimSimplex):
                logger.debug(i)
                data_extreme_points[i + 1] = data[(i + 1, )]
        else: # array
            argmin = data.argmin(axis=0)
            for i in range(self.dimSimplex):
                key = i+1
                data_extreme_points[i+1] = data[argmin[i],:]
        C = {}
        list_base_function_index = [
            i for i in BezierIndex(dim=self.dimSimplex,
                                   deg=self.degree)
        ]
        list_extreme_point_index = [
            i for i in list_base_function_index if count_nonzero(i) == 1
        ]
        for key in list_extreme_point_index:
            index = int(nonzero_indices(key)[0])
            C[key] = data_extreme_points[index + 1]
        for key in list_base_function_index:
            if key not in C:
                C[key] = np.zeros(self.dimSpace)
                for key_extreme_points in list_extreme_point_index:
                    index = int(nonzero_indices(key_extreme_points)[0])
                    C[key] = C[key] + C[key_extreme_points] * (key[index] /
                                                               self.degree)
        return (C)

    def read_control_point(self, filename):
        """Read control point

        Parameters
        ----------
        filename : str(file path and name)
            write data file name

        Returns
        ----------
        c : dict
            control point
        """
        with open(filename, mode="rb") as f:
            c = pickle.load(f)
        return (c)

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

    def write_meshgrid(self, C, filename):
        """Output meshgrid

        Parameters
        ----------
        C : dict
            control point
        filename : str(file path and name)
            write data file name

        Returns
        ----------
        xx_ : numpy.ndarray
            Data to be saved to a text file
        """
        tt_, xx_ = self.meshgrid(C)
        np.savetxt(filename, xx_)
        return (xx_)


if __name__ == '__main__':
    import model
    from itertools import combinations

    DEGREE = 3  # ベジエ単体の次数
    DIM_SIMPLEX = 5  # ベジエ単体の次元
    DIM_SPACE = 5  # 制御点が含まれるユークリッド空間の次元
    NG = 21
    NEWTON_ITR = 20
    MAX_ITR = 30  # 制御点の更新回数の上界

    # input data
    base_index = ['1', '2', '3', '4', '5']
    subsets = []
    for i in range(len(base_index) + 1):
        for c_num in combinations(base_index, i):
            subsets.append(c_num)
    data = {}
    for e in subsets:
        if len(e) == 1:
            data[e] = np.loadtxt('../data/normalized_pf/normalized_5-MED.pf_' +
                                 e[0])
        if len(e) == 5:
            # data[e] = np.loadtxt('data/normalized_5-MED.pf_1_2_3_4_5')
            data[e] = np.loadtxt(
                '../data/normalized_pf/normalized_5-MED.pf_1_2_3_4_5_itr0')

    bezier_simplex = model.BezierSimplex(dimSpace=DIM_SPACE,
                                         dimSimplex=DIM_SIMPLEX,
                                         degree=DEGREE)
    C_init = bezier_simplex.initialize_control_point(data)
    for key in C_init:
        logger.debug("{} {}".format(key, C_init[key]))
    x = bezier_simplex.sampling(C_init, [1, 0, 0, 0])
    logger.debug(x)
    tt, xx = bezier_simplex.meshgrid(C_init)
    logger.debug("{} {}".format(tt.shape, xx.shape))
    bezier_simplex.write_meshgrid(C_init, "sample_mesghgrid")
