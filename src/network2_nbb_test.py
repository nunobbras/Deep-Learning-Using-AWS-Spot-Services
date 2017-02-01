import network2_nbb
import numpy as np

def test_averaging_list_of_lists_of_arrays():
    
    depth = 5
    a = []; b = []; c = []
    for i in np.arange(depth):
#        exec "a" + str(i) + " = np.random.random((3,3))*12"
#        exec "b" + str(i) + " = np.random.random((10,1))*5"
#        exec "c" + str(i) + " = np.random.random((5,5))*20"
        a.append((np.random.random((3,3))*12).round())
        b.append((np.random.random((10,1))*5).round())
        c.append((np.random.random((5,5))*20).round())

    z_list = [[a[i],b[i],c[i]] for i in np.arange(depth)]
    result = []
    result.append(np.average([a[i] for i in np.arange(depth)],axis=0))
    result.append(np.average([b[i] for i in np.arange(depth)],axis=0))
    result.append(np.average([c[i] for i in np.arange(depth)],axis=0))

    assert(network2_nbb.averaging_list_of_lists_of_arrays(z_list) == result)