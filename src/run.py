import numpy as np
import time
import network3_nbb

print "Starting, using " + network3_nbb.theano.config.device

training_data, validation_data, test_data = network3_nbb.load_data_shared()
mini_batch_size = 10    

lmbda = 0.1#lmbda_epochs[0]
epoch = 2#lmbda_epochs[1]

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

#file_name = "./results.txt"
#file_to_write = open(file_name,"wb")
#file_to_write.write(str([net, acc_array, lmbda, epoch]))
#file_to_write.close()

print "results:" + str([net, acc_array, lmbda, epoch])





# import theano
# from theano import function, config, shared, sandbox
# import theano.tensor as T
# import numpy
# import time

# print "ola, entrei no teste e vou usar " + theano.config.device + " e " + theano.config.floatX


# vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
# iters = 1000

# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], T.exp(x))
# print f.maker.fgraph.toposort()
# t0 = time.time()
# for i in xrange(iters):
#     r = f()
# t1 = time.time()
# print 'Looping %d times took' % iters, t1 - t0, 'seconds'
# print 'Result is', r
# if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
#     print 'Used the cpu'
# else:
#     print 'Used the gpu'
