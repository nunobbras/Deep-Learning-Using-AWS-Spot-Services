"""my_func.py
~~~~~~~~~~

"""

def load():
  import numpy as np
  import time

  import network3_nbb

  print "...in load, using " + network3_nbb.theano.config.device

  training_data, validation_data, test_data = network3_nbb.load_data_shared()
  mini_batch_size = 10 

#************************************************************************************
#************************************************************************************
#************************************************************************************
def teste():
  return("green")


def dbl_conv_relu(lmbda_epochs):
    import numpy as np
    import time

    import network3_nbb

    print "...in dbl_conv_relu, using " + network3_nbb.theano.config.device

    training_data, validation_data, test_data = network3_nbb.load_data_shared()
    mini_batch_size = 10    

    lmbda = lmbda_epochs[0]
    epoch = lmbda_epochs[1]
    print "Conv + Conv + FC num %s, relu, with regularization. lambda: %s, epochs: %s" % (0, lmbda, epoch)
    net = network3_nbb.Network([
        network3_nbb.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2), 
                      activation_fn=network3_nbb.ReLU),
        network3_nbb.ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                      filter_shape=(40, 20, 5, 5), 
                      poolsize=(2, 2), 
                      activation_fn=network3_nbb.ReLU),
        network3_nbb.FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=network3_nbb.ReLU),
        network3_nbb.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    acc_array = net.SGD(training_data, int(epoch), mini_batch_size, 0.03, validation_data, test_data, lmbda=lmbda)
    return [acc_array, lmbda, epoch]

#************************************************************************************
#************************************************************************************
#************************************************************************************

def dummy(lmbda_epochs):
    print " ...in dummy " + str(lmbda_epochs)
    tc = time.clock()  
    tt = time.time()
    # #for lmbda in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:    
    #lmbda = lmbda_epochs[0]
    #epochs = lmbda_epochs[1]
    A = np.random.random((lmbda_epochs,lmbda_epochs))
    ev = np.linalg.eigvals(A)
    tcn = time.clock()-tc
    ttn = time.time()-tt
    print "finito do dummy..!"
    return [tcn, ttn, lmbda_epochs]



def test_gpu():

  import theano
  import theano.tensor as T
  import numpy
  import time
  import os

  print " ...in teste e vou usar (novo)" + theano.config.device
  #os.environ['THEANO_FLAGS'] ='floatX=float32,device=gpu*,force_device=True,exception_verbosity=high,nvcc.fastmath=True'
  print os.environ['THEANO_FLAGS']
  print "done"

  vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
  iters = 1000

  rng = numpy.random.RandomState(22)
  x = theano.shared(numpy.asarray(rng.rand(vlen), theano.config.floatX))
  f = theano.function([], T.exp(x))
  print f.maker.fgraph.toposort()
  t0 = time.time()
  for i in xrange(iters):
      r = f()
  t1 = time.time()
  print 'Looping %d times took' % iters, t1 - t0, 'seconds'
  print 'Result is', r
  if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
      print 'Used the cpu'
  else:
      print 'Used the gpu'



def check_theano():
    import theano
    return theano.config.mode, theano.config.device, theano.config.floatX, theano.config.base_compiledir


