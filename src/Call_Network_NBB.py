# -*- coding: utf-8 -*-

# NBB - Network 1 - Analysis and Testing
# Going to analyze Network 1, 2 and 3 from Nielsen and show how it works.
# Creating the same files as Nielsen
# Adding paths

#%load_ext autoreload
#%autoreload 2



import sys
#rever path
sys.path.append("/Users/re_nbb/Dropbox/Research_Code/Theano/Nielsen&Theano/ORIGINAL_neural-networks-and-deep-learning/src")

import mnist_loader
import numpy as np
import nbbplotly
import network_nbb



training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#m = training_data[5]
#M = np.reshape(m[0],[28,28])
#M1 = M[::-1,:]
#nbbplotly.plot(M1)
#y = m[1]

#normal training
net = network_nbb.Network([784, 30, 10])
net.SGD(training_data, 3, 10, 3.0, test_data=test_data)

nbbplotly.plot(a)

#smaller values training

reload(network_nbb)
training_data_sub = training_data[:1000]
net = network_nbb.Network([784, 30, 10])
net.SGD(training_data_sub, 10, 10, 3.0, test_data=test_data, init_method = "old")




#Call network_2
# Treino para analise dos resultados usando uma NN com duas unidades por layer.
import sys
#review path

import mnist_loader
import numpy as np
import nbbplotly
import network2_nbb

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

reload(nbbplotly)
reload(network2_nbb)
training_data_sub = training_data[:1000]
net = network2_nbb.Network([784, 2, 2, 2, 2, 10], cost=network2_nbb.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 3, 10, 0.1, lmbda = 5.0, evaluation_data=validation_data,monitor_evaluation_accuracy=True)





#Call network_3
#

import sys

import mnist_loader
import numpy as np
import nbbplotly
import network3_nbb

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()




reload(network3_nbb)
layer_numbers = [784, 5, 5, 10]

#gera as layers
layers = [network3_nbb.FullyConnectedLayer(layer_numbers[n],layer_numbers[n+1]) for n in range(layer_numbers.__len__()-2)]
net = network3_nbb.Network(layers, 10)

net.SGD(training_data, 30, 10, 0.1, validation_data, test_data)
net.SGD(training_data, 30, 10, 0.1, validation_data, test_data)



#Call conv

plt = conv.plot_errors(error_locations, erroneous_predictions)
plt.savefig("ensemble_errors.png")
# plot the filters learned by the first of the nets just trained
plt = conv.plot_filters(nets[0], 0, 5, 4)
plt.savefig("net_full_layer_0.png")
plt = conv.plot_filters(nets[0], 1, 8, 5)
plt.savefig("net_full_layer_1.png")


#Call conv small test;

import sys

import mnist_loader

#import nbbplotly

import network3_nbb
import numpy as np
import pickle

reload(network3_nbb)
reload(conv)

#Se for GPU

import numpy as np
import my_func


training_data, validation_data, test_data = network3_nbb.load_data_shared()
lmbda = .1
epochs = 5
lmbda_epochs_array = [lmbda,epochs]
my_func.set_GPU(True)
ar = my_func.dbl_conv_relu (lmbda_epochs_array, False)

results[0][0].SGD = None
results[0][0].test_mb_predictions = None

f = open("./results/GPU_master/results/res_1_CPU.pck", "wb")
s = pickle.dump(results,f)
f.close()

#Parallel call

import my_func
#reload(conv)
from IPython.parallel import Client
rc = Client()
dview = rc[:]

#lmbda = [0.0, 0.00001]#, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
#epochs = [1, 1]#, 1, 1, 1, 1, 1, 1, 1]
nets = []
acc_arrays = []
dview.execute('import os')
dview.execute('os.chdir("/home/ubuntu/Dropbox/Research_Code/ML/Nielsen/My_Code")')
#dview.execute('sys.path.append("/home/ubuntu/Dropbox/Research_Code/ML/Nielsen/My_Code")')
dview.execute('import my_func')
dview.execute('reload(my_func)')
dview.apply(my_func.network3_nbb.set_GPU,True)

async = dview.map_async(conv.dbl_conv_relu, lmbda, epochs)
print('async output:')
async.display_outputs()
print async.get()

nets.append(net)
acc_arrays.append(acc_array)


dv1 = rc[1]
dv2 = rc[2]
dv3 = rc[3]
dv4 = rc[4]

lmbda = 0.0
epochs = 1
dview.targets = [2]
dview.block = False
ar1 = dv1.apply(conv.dbl_conv_relu)
ar2 = dv2.apply(conv.dbl_conv_relu)
ar3 = dv3.apply(conv.dbl_conv_relu)
ar4 = dv4.apply(conv.dbl_conv_relu)
ar.get()


%px config --targets 0:2
%px print 'hi'


######### NOVO
%load_ext autoreload
%autoreload 2



import my_func
from IPython.parallel import Client
import numpy as np

rc = Client()
dv = rc[1]
rc.ids
dv

dv.block = False
lmbda = np.arange(1)*.1
epochs = (np.zeros(1)*0+2).astype(int)
lmbda_epochs_array = np.concatenate((lmbda[:,np.newaxis], epochs[:,np.newaxis]), axis=1)

ar = dv.map_async(my_func.dummy, lmbda_epochs_array); see(ar);
ar = dv.map_async(my_func.dbl_conv_relu, lmbda_epochs_array); see(ar);

ar = my_func.dummy(lmbda_epochs_array);

######## NOVO - Cluster GPU - Enviando os comandos, com objectos locais


from IPython.parallel import Client
import numpy as np
import time

rc = Client()
dv = rc[0,1,20,21,40,41,50,51,65,66,80,81,100,101,110,111]
rc.ids
dv.block = True
dv.activate()

%px %load_ext autoreload
%px %autoreload 2

#%px import os
#%px os.chdir("/home/ubuntu/Dropbox/Research_Code/ML/Nielsen/My_Code")

%px cd /home/ubuntu/Dropbox/Research_Code/ML/Nielsen/My_Code
%px import my_func
%px import numpy as np
%px import os
%px import subprocess

%px %env LD_LIBRARY_PATH=/usr/local/cuda/lib64
%px %env THEANO_FLAGS = floatX=float32,device=gpu0
#%px %env THEANO_FLAGS = floatX=float32,device=gpu0,force_device=True,exception_verbosity=high

#%px reload(my_func)

#%px lmbda_epochs = (0.01,1)
#ar = %px my_func.dbl_conv_relu(lmbda_epochs)
#%px my_func.test_gpu()

# %px os.system("python run.py > olecas.txt")
# %px del theano

t0 = time.time()
%px subprocess.check_output("python run.py; exit 0", stderr=subprocess.STDOUT, shell=True)
t1 = time.time()
T = t1-t0
print T

#%px subprocess.check_output( "ls non_existent_file; exit 0", stderr=subprocess.STDOUT, shell=True)
#%px subprocess.check_output("echo $LD_LIBRARY_PATH; exit 0", stderr=subprocess.STDOUT, shell=True)

#%px subprocess.check_output("echo $PATH; exit 0", stderr=subprocess.STDOUT, shell=True)
#%px os.environ['SHELL']

######MAis uma tentativa sem %px


from IPython.parallel import Client
from IPython.parallel import CompositeError
CompositeError.tb_limit = 1
import numpy as np

rc = Client()
dv = rc[1]
rc.ids
dv.block = False

dv.execute("cd /home/ubuntu/Dropbox/Research_Code/ML/Nielsen/My_Code", block=True)
dv.execute("%env LD_LIBRARY_PATH=/usr/local/cuda/lib64")
dv.execute("%env THEANO_FLAGS = floatX=float32,device=gpu0")
dv.execute("%load_ext autoreload")
dv.execute("%autoreload 2")

dv.targets = [1]
a = dv.execute("%env THEANO_FLAGS")
a.result;a.display_outputs();a.stdout 

dv.execute("import numpy as np", block=True)
dv.execute("import my_func", block=True)
a = dv.execute("import theano", block=True)
a.result;a.display_outputs()
dv.execute("import theano.tensor as T", block=True)
dv.execute("import numpy", block=True)
dv.execute("import time", block=True)
dv.execute("import os", block=True)

a = dv.execute("theano.config.force_device = True", block=True)
a = dv.execute("print theano.config", block=True)
a = dv.execute("print os.getenv('THEANO_FLAGS')", block=True)
a.result;a.display_outputs()




a = dv.execute("os.environ['THEANO_FLAGS'] ='floatX=float32,device=gpu0,Force_device=True,mode=DebugMode,exception_verbosity=high'")
a = dv.execute("print os.environ['THEANO_FLAGS']", block=True)
a.result;a.display_outputs()

a = dv.execute("vlen = 10 * 30 * 768")

a = dv.execute("rng = numpy.random.RandomState(22)", block=True)
a.result;a.display_outputs()
a = dv.execute("x = theano.shared(numpy.asarray(rng.rand(vlen), theano.config.floatX))")
a.result;a.display_outputs()
a = dv.execute("f = theano.function([], T.exp(x))")
a.result;a.display_outputs()
a = dv.execute("A = 10 if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]) else 20")
a.result;a.display_outputs();
a = dv.execute("print A")
a.result;a.display_outputs()
a = dv.execute("my_func.test_gpu()")
a.result;a.display_outputs()




########FUNC see(ar)

import sys
import time

def see(ar, dt=1, truncate=120):
    while not ar.ready():
        stdouts = ar.stdout
        if not any(stdouts):
            continue
        # clear_output doesn't do much in terminal environments
        print "\n"  * 60                  
        print '-' * 30
        print "%.3fs elapsed" % ar.elapsed
        print ""
        for eid, stdout in enumerate(ar.stdout):
            if stdout:
                print "[ stdout %2i ] %s" % (eid, stdout[-truncate:].rstrip('\n'))
#        ar.stdout = []
        time.sleep(dt)
    print '-' * 30
    print 'FINISHED:'
    print '-' * 30
    for eid, stdout in enumerate(ar.stdout):
        if stdout:
            print "[ stdout %2i ] %s" % (eid, stdout)    



dv.block = True
dv.execute('import os')
dv.execute('os.system("sudo fallocate -l 4G /swapfile")')
dv.execute('os.system("sudo chmod 600 /swapfile")')
dv.execute('os.system("sudo mkswap /swapfile")')
dv.execute('os.system("sudo swapon /swapfile")')




def instantiate_empty_variable_code(self, var):
    assert var.dtype in (np.float32, 'float32')
    name = self.local_name(var)
    code = '%s = theano.sandbox.cuda.CudaNdarray.zeros(%s);'%(
            name, str(var.shape))
    return code





