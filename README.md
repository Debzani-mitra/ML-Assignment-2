# ML-Assignment-2
Gram-Schmidt process-Assignment submitted for online Machine learning Math course offered by Imperial College London 

import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14 # That's 1×10⁻¹⁴ = 0.00000000000001

# performs the Gram-Schmidt procedure for 4 basis vectors.

def gsBasis4(A) :
    B = np.array(A, dtype=np.float_) 
    # normalising.
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    # subtracting overlap.
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]
    # when B[:, 1] is linearly independant of B[:, 0], normalise.
    if la.norm(B[:, 1]) > verySmallNumber :
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else :
        B[:, 1] = np.zeros_like(B[:, 1])
    # column 2.
    B[:,2]=B[:,2]-B[:,2] @ B[:,0] * B[:,0]
    B[:,2]=B[:,2]-B[:,2] @ B[:,1] * B[:,1]
    # normalising.
    if la.norm(B[:,2])>verySmallNumber:
        B[:,2]=B[:,2]/la.norm(B[:,2])
    else:
        B[:,2]=np.zeros_like(B[:,2])
    # column 3.
    B[:,3]=B[:,3]-B[:,3] @ B[:,0] * B[:,0]
    B[:,3]=B[:,3]-B[:,3] @ B[:,1] * B[:,1]
    B[:,3]=B[:,3]-B[:,3] @ B[:,2] * B[:,2]
    # normalising if possible.
    if la.norm(B[:,3])>verySmallNumber:
        B[:,3]=B[:,3]/la.norm(B[:,3])
    else:
        B[:,3]=np.zeros_like(B[:,3])
    return B

# generalising the procedure.

def gsBasis(A) :
    B = np.array(A, dtype=np.float_) 
    # Looping over all vectors
    for i in range(B.shape[1]) :
        # Inside looping over all previous vectors, j, to subtract.
        for j in range(i) :
            B[:, i] = B[:,i]-B[:,i] @ B[:,j] * B[:,j]
        # normalising
        if la.norm(B[:,i])>verySmallNumber:
            B[:,i]=B[:,i]/la.norm(B[:,i])
        else:
            B[:,i]=np.zeros_like(B[:,i])
    return B

def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0)) 
