
#Deep Learning over GPU Servers at AWS

This code was used to produce the article "Clustering Low-Cost, Cloud-Based Servers to Solve Intensive, Parallel Computations" by Nuno B. Bras and Goncalo Valadao from Instituto de Telecomunicac ̧o ̃es (IT) and Universidade Auto ́noma de Lisboa (UAL) Lisboa, Portugal.

This paper advocates the usage of available cloud based services for intensive parallel floating point computations, based on clusters of servers installed with Graphical Processing Units (GPUs), in order to run low-cost, High Performance Computing (HPC) tasks.
It is described a cluster of multiple servers installed with GPU units and running open-source software which works as an easy to scale, low cost platform with centralized Master-Slave control, able to turn on/off the Slaves (GPU server machines) as needed.

###Used Technologies:

- **Amazon Web Services (AWS) with AWS Spot Services**
AWS Spot Services offer a new modality which presents much lower prices. Essentially it consists of serving unused processing power and its price can be much lower than of a normal, reserved, GPU server instances [6].

- **StarCluster  http://star.mit.edu/cluster/**
We define a master-slave structure for the employed cluster, where the master server is responsible for coordinating slave servers and making data persistent across them.

- **iPython Parallel module**
Regarding our application with a master and slaves architecture, it is possible to write in python the execution code but also the master scripts to control slaves

- **ZeroMQ**
Send direct iPython parallel messages from the master to any set of engines instances running in the slaves.

- **Theano**
 Theano is always used, even when GPUs are not available, and Theano itself chooses what should be used: CPU or GPUs.

 
###Underlying Problem

The underlying problem is a character classification problem, over the MNIST dataset. This problem was often used to benchmark algorithms because it was already deeply studied. The MNIST has a training set of 60 000 characters, a testing set of 10 000 characters, having each image 28 × 28 pixels.
Here, the problem is solved using a deep neural network described in [14], composed of the following layers:

- 2 layers, each composed of a convolutional layer and a pool layer (2×2) (half size image reduction in each Pool layer). This takes as input 784 pixels, and at the end of the last layer, one have 100 values;
- 1 Fully Connected Layer using a ReLU activation func- tion, input layer size: 25 values and an output of 25 values;
- 1 Final Softmax Layer giving statistical meaning to the obtained results, with a 25 input arrays and with 10 output results (one for each numerical digit);

where a similar Network was used to classify the ImageNet Dataset. The resulting optimization problem is regularized using an L1 kernel, although other methods could be used such as Dropout techniques.


###



