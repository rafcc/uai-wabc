import numpy as np
from math import factorial
import os
import ot
import copy
import matplotlib
import trainer
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# how to download it is
# $ pip install pot

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
def calc_diff(mu_pos,mu_pre):
    return(np.linalg.norm(mu_pos-mu_pre)/np.linalg.norm(mu_pre))
def get_normalized_data(name):
    path = '../data/raw/{}.pf'.format(name)
    Data = Dataset(path)
    N = Normalizer(Data.data)
    ndata = N.normalize()
    return ndata

def i_list(N, M):
    ''' N次、M次元のBazier indexを返す '''
    l = []
    for n in range(10**M, 2*10**M):
        i = [int(x) for x in list(str(n))]
        i = i[1:]
        if sum(i) == N:# sum(i) <= N でよい??
            l.append(i)
    return l

def uniform_on_simplex(d):
    z = np.sort(np.random.uniform(0.0, 1.0, d))
    return np.append(z, 1.0) - np.append(0.0, z)

def sample_simplex(bs, d_simplex):
    ''' bs個のサンプルをd_simplex次元の単体から一様にサンプルする
        （一旦[0,1]**d_simplex から一様分布でサンプルし、合計が1を超えないやつだけbs個とる、という内容なので
        　場合によってはbs個も作れなくて最後の行でエラーが出るかも。）'''
    t = np.zeros(shape=(bs, d_simplex+1))
    for i in range(bs):
        t[i] = uniform_on_simplex(d_simplex)
    return t

def make_image(y, X_train_cpu):
    dim = y.shape[1]
    Gs_cpu = y
        #
    fig = plt.figure(figsize=(dim*3, dim*3))
    for x in range(dim):
        for y in range(dim):
            ax = fig.add_subplot(dim, dim, x+1+dim*y)
            X, Y = X_train_cpu[:,x:x+1], X_train_cpu[:,y:y+1]
            ax.set_xlim(np.min(X)-.5, np.max(X)+.5)
            ax.set_ylim(np.min(Y)-.5, np.max(Y)+.5)
            ax.scatter(Gs_cpu[:,x:x+1], Gs_cpu[:,y:y+1], alpha=0.5, color='green', marker='.',)
            ax.scatter(X, Y, alpha=0.5, color='gray', marker='.', label='True data')


def multinom(d):
    ''' d = [d1, d2, ..., dM] '''
    num = np.math.factorial(np.sum(d))
    denom = np.vectorize(np.math.factorial)(d) # from https://teratail.com/questions/264418
    denom = np.prod(denom)
    return num/denom

class BezierSimplex():
    def __init__(self, degree, d_simplex, d_out):
        self.d_simplex = d_simplex
        self.degree = degree
        self.d_out = d_out
        self.degrees_list_np = np.array(i_list(degree, d_simplex+1))
        print(self.degrees_list_np)
        self.multinomials_np = np.array([multinom(d) for d in self.degrees_list_np]).reshape(1, -1, 1, 1)
        self.mu_np = np.zeros((len(self.degrees_list_np), d_out))
        self.sigma_np = np.array([np.identity(d_out)*0.1 for _ in range(len(self.degrees_list_np))])
        self.number_of_updates = 0


    def sample_control_points_np(self):
        ''' mu_list_np = [mu_{d1}, mu_{d2}..., ] '''
        control_points_list = []
        for ind in range(len(self.degrees_list_np)):
            mu = self.mu_np[ind]
            sigma = self.sigma_np[ind]
            p_d = np.random.multivariate_normal(mean=mu, cov=sigma)
            control_points_list.append(p_d)
        return np.array(control_points_list)

    def __call__(self, t_np, control_points_np=None):
        ''' t_np.shape                 = (sample_size, -, d_simplex, -)
            control_points_np.shape    = (-, len(self.degrees_list_np), -, d_out)
            self.degrees_list_np.shape = (-, len(self.degrees_list_np), d_simplex, -)
            --> (sample_size, len(self.degrees_list_np), d_simplex, d_out)
        '''
        t_np = np.expand_dims(t_np, axis=1)
        t_np = np.expand_dims(t_np, axis=3)

        if control_points_np is None:
            control_points_np = self.sample_control_points_np()
        control_points_np = np.expand_dims(control_points_np, axis=0)
        control_points_np = np.expand_dims(control_points_np, axis=2)

        degrees_list_np = np.expand_dims(self.degrees_list_np, axis=0)
        degrees_list_np = np.expand_dims(degrees_list_np,      axis=3)
        t_pow_d = np.prod(t_np**degrees_list_np, keepdims=True, axis=2)
        # --> (sample_size, len(self.degrees_list_np), 1, d_out)
        bt = np.sum(self.multinomials_np*t_pow_d*control_points_np, keepdims=True, axis=1)
        # --> (sample_size, 1, 1, d_out)
        return np.squeeze(bt)

def emd(x, y):
    '''
    Earth-Mover's Distance (Wasserstein?)

    https://pythonot.github.io/index.html
    x.shape = (N, d_out)
    y.shape = (N, d_out)
    '''
    N = x.shape[0]
    N_= y.shape[0]
    M = ot.dist(x, y,metric='sqeuclidean')
    #M /= M.max()
    a, b = np.ones((N,))/N, np.ones((N_,))/N_
    return np.sqrt(ot.emd2(a, b, M))

def gd(dd1,dd2):
    """
    dd1 input sample
    dd2 estimated bezier simplex sample
    """
    gd =0
    for i in range(dd2.shape[0]):
        d2 = dd2[i,:]#.reshape(1, dd2.shape[1])
        #print(dd1, d2)
        tmp = dd1 - d2
        norm = np.linalg.norm(tmp,1,axis=1)
        v = np.min(norm)
        gd += v
    return gd/dd2.shape[0]

def igd(dd1,dd2):
    """
    dd1 input sample
    dd2 estimated bezier simplex sample
    """
    igd = 0
    for i in range(dd1.shape[0]):
        d1 = dd1[i,:]
        tmp = dd2 - d1
        norm = np.linalg.norm(tmp,1,axis=1)
        v = np.min(norm)
        igd += v
    return igd/dd1.shape[0]

def distance_mean(b, data, distance, trial=10):
    return np.mean([distance(data, b(sample_simplex(bs=len(data), d_simplex=b.d_simplex))) for _ in range(trial)])

class ABC_estimator():
    def __init__(self, data, bezierSimplex, batch_size, distance):
        self.data = data
        self.distance = distance # function
        self.N = batch_size#data.shape[0]
        self.bezierSimplex = bezierSimplex
        print(self.bezierSimplex.d_out,self.bezierSimplex.d_simplex,self.bezierSimplex.degree)
        self.borges= trainer.BorgesPastvaTrainer(dimSpace=self.bezierSimplex.d_out,
                                            dimSimplex=self.bezierSimplex.d_simplex+1,
                                            degree = self.bezierSimplex.degree)
        self.reset2()

    def step2(self, delta):
        control_points_np = self.bezierSimplex.sample_control_points_np()
        t_np = sample_simplex(bs=self.N, d_simplex=self.bezierSimplex.d_simplex)
        y = self.bezierSimplex(t_np, control_points_np=control_points_np)
        np.random.shuffle(self.data)
        accept = self.distance(self.data[:self.N], y) <= delta
        if accept:
            self.control_points.append(control_points_np)
        return accept

    def initialize(self):
        C = self.borges.bezier_simplex.initialize_control_point(
                data=self.data)
        print("aaaa",self.bezierSimplex.degrees_list_np)
        print("control",C)
        for ind in range(len(self.bezierSimplex.degrees_list_np)):
            deg = tuple(self.bezierSimplex.degrees_list_np[ind])
            #print("iii",ind,deg,C[deg].T.shape,np.mean(self.data, axis=0).shape)#,np.mean(self.data, axis=0))
            self.bezierSimplex.mu_np[ind, :] = C[deg]#np.mean(self.data, axis=0)
    def update2(self):
        control_points = np.array(self.control_points) # (acceptance, len(bezierSimplex.degrees_list_np), dim_out)
        for ind in range(len(self.bezierSimplex.degrees_list_np)):
            self.bezierSimplex.mu_np[ind, :] = np.mean(control_points[:, ind, :], axis=0)
            self.bezierSimplex.sigma_np[ind, :] = np.cov(control_points[:, ind, :].T)
        #print(self.bezierSimplex.mu_np)
        self.reset2()

    def reset2(self):
        self.control_points = []

    def execute3(self, M, delta=0.1, progress_bar=True):
        if progress_bar:
            print("progress(#=10%):", end="")
        loop = 0
        flg = 0
        while len(self.control_points) <= M:
            accept = self.step2(delta=delta)
            if accept and progress_bar:
                if len(self.control_points)%(M//10)==0:
                    print("#", end="")
            loop += 1
            if loop >= 10**5:
                flg = 1
                print("reach the maximum iteration")
                break
        return(loop,flg)
        print(" ABC finished, number of acceptances =", len(self.control_points))

def run_ABC(ndata,result_dir, N_update=30, M=500, degree=3, discount_rate=0.8, show=True,
           b=None, batch_size=100,seed=0,tolerance=10**(-4), distance=emd):
    create_directory(result_dir)
    np.random.seed(seed)
    np.random.shuffle(ndata)
    d_out = ndata.shape[1]
    d_simplex = d_out-1
    if b is None:
        b = BezierSimplex(degree=degree, d_simplex=d_simplex, d_out=d_out)
    abc_estmator = ABC_estimator(ndata, b, batch_size, distance=distance)
    print(ndata.shape)
    abc_estmator.initialize()
    t_np = sample_simplex(bs=1000, d_simplex=d_simplex)
    D = distance_mean(b, ndata[:1000], distance=distance, trial=10)
    #D = np.mean([distance(ndata, b(t_np, control_points_np=b.mu_np)) for _ in range(1)])
    info = {}
    info["max_iter"] = 0
    info["max_cov"] = 0

    for g in range(N_update):
        mu_pre = copy.copy(b.mu_np)
        print(g,'-th update')
        delta=discount_rate*D
        print(' {} ='.format(str(distance).split(" ")[1]), D,
              '\n delta = {}*{} = '.format(discount_rate, str(distance).split(" ")[1]), delta)
        itr,flg = abc_estmator.execute3(M=M, delta=delta)

        if flg==1:
            info["max_iter"] = 1
            y = b(t_np, control_points_np=b.mu_np)
            make_image(y, ndata[:1000])
            plt.savefig(result_dir+'/'+str(g)+'.png')
            plt.close()
            break
        else:
            abc_estmator.update2()
            l= [np.max(np.linalg.eig(b.sigma_np[i])[0]) for i in range(len(b.sigma_np))]
            print("eig max",np.max(l))
            if np.max(l) <= 10**(-3):
                info["max_cov"] = 1
                y = b(t_np, control_points_np=b.mu_np)
                make_image(y, ndata[:1000])
                plt.savefig(result_dir+'/'+str(g)+'.png')
                plt.close()
                break
            else:
                if g%10==0:
                    y = b(t_np, control_points_np=b.mu_np)
                    make_image(y, ndata[:1000])
                    plt.savefig(result_dir+'/'+str(g)+'.png')
                    plt.close()
                D_pre = D
                #D = np.mean([distance(ndata, b(t_np, control_points_np=b.mu_np)) for _ in range(1)])
                D = distance_mean(b, ndata[:1000], distance=distance, trial=10)
                # terminate condition
                mu_pos = copy.copy(b.mu_np)
                diff = np.abs(D_pre-D)
                #diff = calc_diff(mu_pos=mu_pos,mu_pre=mu_pre)
                print("diff",diff)
            """
            if diff <= tolerance:
                y = b(t_np, control_points_np=b.mu_np)
                make_image(y, ndata[:1000])
                plt.savefig(result_dir+'/'+str(g)+'.png')
                plt.close()
                break
            """
    y = b(t_np, control_points_np=b.mu_np)
    make_image(y, ndata[:1000])
    plt.savefig(result_dir+'/'+str(g)+'.png')
    plt.close()
    return b,mu_pos,g+1,info
if __name__ == '__main__':
    import data
    DATA_DIR = '../data/preprocessed'
    RES_DIR = '../result'
    DATA_NAME = '3-MED'#argvs[1]  # '3-MED'
    DATA_ALL = data.Dataset(DATA_DIR + '/' + DATA_NAME + '.pf').values

    params = {"ndata":DATA_ALL[:100],
              "N_update":50,
              "M":100, "degree":3, "discount_rate":0.8, "show":True,
              "tolerance":10**(-3),
              "b":None, "batch_size":100,"seed":0}
    result_dir = RES_DIR+"/data.{}".format(DATA_NAME)
    for key in ["M","degree","discount_rate","batch_size","tolerance","seed"]:
        result_dir = result_dir+","+key+".{}".format(params[key])
    params["result_dir"] = result_dir
    run_ABC(**params)
