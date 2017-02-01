"""loader.py
~~~~~~~~~~

loader do cluster
e funcoes auxiliares
"""

%load_ext autoreload
%autoreload 2

import my_func
from IPython.parallel import Client
import numpy as np

rc = Client()
dv = rc[1,2,3,4]
rc.ids
dv
dv.block = True

dv.execute('%load_ext autoreload')
dv.execute('%autoreload 2')
dv.execute('import os')
dv.execute('os.chdir("/home/ubuntu/Dropbox/Research_Code/ML/Nielsen/My_Code")')
dv.execute('import my_func')
dv.execute('reload(my_func)')
dv.apply(my_func.network3_nbb.set_GPU,False)

dv.execute('os.system("rm my_func.pyc")')
dv.execute('os.system("rm network3_nbb.pyc")')

#ar = dv.map_async(my_func.dbl_conv_relu,[[0.01,1],[0.02,1]]); see(ar);
#ar.display_outputs()
dv.block = False
lmbda = np.arange(4)*.1
epochs = (np.zeros(4)*0+2).astype(int)
lmbda_epochs_array = np.concatenate((lmbda[:,np.newaxis], epochs[:,np.newaxis]), axis=1)


def start_dummy_job():
    ar = dv.map_async(my_func.dummy, lmbda_epochs); see(ar);
    see(ar);



def start_job():
    ar = dv.map_async(my_func.dbl_conv_relu, lmbda_epochs_array); 
    see(ar);


def see(ar, dt=1, truncate=1000):
    while not ar.ready():
        clear_output()
        print '-' * 30
        print "%.3fs elapsed" % ar.elapsed
        for out in ar.stdout: print out
        time.sleep(dt)

    print 'FINISHED:' + '-' * 30
    ar.display_outputs()
    print(ar.result)