# ML-Assignment-2
Gram-Schmidt process-Assignment submitted for online Machine learning Math course offered by Imperial College London 
import numpy as np

# function that goes through the matrix replacing each row in order turning it into echelon form.

def isSingular(A) :
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    try:
        fixRowZero(B)
        fixRowOne(B)
        fixRowTwo(B)
        fixRowThree(B)
    except MatrixIsSingular:
        return True
    return False

# defines the error flag for when things go wrong if the matrix is singular.
class MatrixIsSingular(Exception): pass

# For Row Zero, first element needs to be equal to 1.

def fixRowZero(A) :
    if A[0,0] == 0 :
        A[0] = A[0] + A[1]
    if A[0,0] == 0 :
        A[0] = A[0] + A[2]
    if A[0,0] == 0 :
        A[0] = A[0] + A[3]
    if A[0,0] == 0 :
        raise MatrixIsSingular()
    A[0] = A[0] / A[0,0]
    return A

# then sub-diagonal elements to zero, i.e. A[1,0].

def fixRowOne(A) :
    A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        A[1] = A[1] + A[2]
        A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        A[1] = A[1] + A[3]
        A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        raise MatrixIsSingular()
    A[1] = A[1] / A[1,1]
    return A
# sub-diagonal elements of row two to zero if not zero
def fixRowTwo(A) :
    A[2]=A[2]-A[2,0]*A[0]
    A[2]=A[2]-A[2,1]*A[1]
    if A[2,2] == 0 :
        A[2]=A[2]+A[3]
        A[2]=A[2]-A[2,0]*A[0]
        A[2]=A[2]-A[2,1]*A[1]     
    if A[2,2] == 0 :
        raise MatrixIsSingular()
    A[2]=A[2]/A[2,2]
    return A

# sub-diagonal elements of row three to zero if not zero
def fixRowThree(A) :
    A[3]=A[3]-A[3,0]*A[0]
    A[3]=A[3]-A[3,1]*A[1]
    A[3]=A[3]-A[3,2]*A[2]
    if A[3,3]==0:
        raise MatrixIsSingular()
    A[3]=A[3]/A[3,3]
    return A
