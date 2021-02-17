import math
from multiprocessing import Process
import time
import data
import trainer
import abc_trainer
import model
import yaml
import numpy as np
import random
import gd_igd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
def make_image(X_trn,X_val,X_pred):
    dim = X_trn.shape[1]
    fig = plt.figure(figsize=(dim*3, dim*3))
    for x in range(dim):
        for y in range(dim):
            ax = fig.add_subplot(dim, dim, x+1+dim*y)
            X, Y = X_trn[:,x:x+1], X_trn[:,y:y+1]
            ax.set_xlim(np.min(X)-.5, np.max(X)+.5)
            ax.set_ylim(np.min(Y)-.5, np.max(Y)+.5)
            ax.scatter(X_pred[:,x:x+1], X_pred[:,y:y+1], alpha=0.5, color='blue', marker='.', label='Bezier simplex')
            ax.scatter(X_val[:,x:x+1], X_val[:,y:y+1], alpha=0.5, color='gray', marker='.', label='True data')
            ax.scatter(X_trn[:,x:x+1], X_trn[:,y:y+1], alpha=1, color='green', marker='.', label='Training data')
            ax.legend()


def sampling(d,n,seed):
    random.seed(seed)
    s = [i for i in range(d.shape[0])]
    s_=random.sample(s,n)
    return(d[s_,:])

def experiments(n,m,sigma,data_dir,data_name,degree,dimspace,dimsimplex,
                               result_dir,seed):
    """
    conduct experiments with synthetic data

    Parameters
    ----------
    n : int
        number of sample points to be trained
    sigma: float
        variance
    degree : int
        max degree of bezier simplex fittng
    dimspace : int
        the number of dimension of the Eucledian space
        where the bezier simplex is embedded
    dimsimplex : int
        the number of dimension of bezier simplex
    result_dir: str
        where to output results
    """
    num_pred = 1000

    # data generation class
    data_all = data.Dataset(data_dir+'/'+data_name+'.pf').values
    data_tst = data_all
    data_tst_array = np.array(data_tst)

    data_trn = sampling(d=data_all,n=n,seed=seed)
    data_trn_array = np.array(data_trn)

    print(data_trn_array[0:10,:])
    epsilon = np.random.multivariate_normal([0 for i in range(dimspace)],
                                             np.identity(dimspace)*(sigma**2),n)
    print(data_trn_array)
    data_trn_array = data_trn_array + epsilon
    borges_pastva_trainer = trainer.BorgesPastvaTrainer(dimSpace=dimspace,
                                                        dimSimplex=dimsimplex,
                                                        degree = degree)
    C_init = borges_pastva_trainer.bezier_simplex.initialize_control_point(
        data=data_trn_array)
    print("initial control",C_init)
    params = {"ndata":data_trn_array,
              "N_update":50,
              "M":m, "degree":degree, "discount_rate":0.8, "show":True,
              "tolerance":10**(-4),
              "b":None, "batch_size":100,"seed":seed}
    print(result_dir,data_name,sigma)
    result_dir = result_dir+"/data."+data_name+",n."+str(n)+",sigma."+str(sigma)
    for key in ["M","degree","discount_rate","batch_size","tolerance"]:
        result_dir = result_dir+","+key+".{}".format(str(params[key]))
    result_dir = result_dir+"/seed.{}".format(params["seed"])

    result_abc = result_dir+"/abc"
    params["result_dir"] = result_abc
    start_abc = time.time()
    b,mu_abc,iter,info = abc_trainer.run_ABC(**params)
    time_abc = time.time()-start_abc


    start_borges = time.time()
    C_init = borges_pastva_trainer.bezier_simplex.initialize_control_point(
        data=data_trn_array)
    result_borges = result_dir+"/borges"
    params_borges = {"data":data_trn_array,"data_val":data_tst,"C_init":C_init,
                     "indices_fix":[],"result_dir":result_borges,
                     "tolerance":10**(-3),"max_iteration":100,
                     "flag_write_meshgrid":0}
    C = borges_pastva_trainer.train(**params_borges)
    time_borges = time.time() - start_borges

    # validate
    t_val_np = abc_trainer.sample_simplex(bs=num_pred, d_simplex=b.d_simplex)
    pred_abc = b(t_np=t_val_np,control_points_np=mu_abc)
    pred_borges = borges_pastva_trainer.bezier_simplex.sampling_array(c=C,t=t_val_np)

    gd_abc,igd_abc = gd_igd.calc_gd_igd(dd1=data_tst_array, dd2=pred_abc)
    gd_borges,igd_borges = gd_igd.calc_gd_igd(dd1=data_tst_array, dd2=pred_borges)

    # output result
    result_dict = {}
    data_dict = {}
    result_dict["borges"] = {"gd":"{:.5f}".format(gd_borges),
                            "igd":"{:.5f}".format(igd_borges),
                             "time":"{:.5f}".format(time_borges)}
    result_dict["abc"] = {"gd":"{:.5f}".format(gd_abc),
                          "igd":"{:.5f}".format(igd_abc),
                          "itr":"{:}".format(iter),
                          "time":"{:.5f}".format(time_abc),
                          "info":info}
    data_dict["train"] = data_trn_array.tolist()
    data_dict["pred"] = {"abc":pred_abc.tolist(),"borges":pred_borges.tolist()}

    with open(result_dir+"/result.yaml","w") as wf:
        yaml.dump(result_dict,wf, default_flow_style=False)
    with open(result_dir+"/data.yaml","w") as wf:
        yaml.dump(data_dict,wf, default_flow_style=False)

    make_image(X_trn=data_trn_array,X_pred=pred_abc,X_val=data_tst_array)
    plt.savefig(result_dir+'/abc.png')
    plt.close()

    make_image(X_trn=data_trn_array,X_pred=pred_borges,X_val=data_tst_array)
    plt.savefig(result_dir+'/borges.png')
    plt.close()
if __name__=='__main__':
    def main(arg):
        jobs = []
        for seed in range(20):
            jobs.append(Process(target=experiments, args=tuple(arg+[seed])))
        start_ = time.time()
        for j in jobs[0:10]:
            j.start()
        for j in jobs[0:10]:
            j.join()
        for j in jobs[10:20]:
            j.start()
        for j in jobs[10:20]:
            j.join()
        elapsed_time = time.time()-start_
    for n in [150,100,50]:
        for m in [100]:#,200,400]:#,200,400]:
            for sigma in [0.1,0.05,0]:
                main(arg=[n,m,sigma,"../data/","Schaffer",3,2,2,"../results"])
                main(arg=[n,m,sigma,"../data/","3-MED",3,3,3,"../results"])
                main(arg=[n,m,sigma,"../data/","Viennet2",3,3,3,"../results"])
                main(arg=[n,m,sigma,"../data/","5-MED",3,5,5,"../results"])
