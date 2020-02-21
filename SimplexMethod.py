import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv 


'''
In mathematical optimization, Dantzig's simplex algorithm (or simplex method)
is a popular algorithm for linear programming. The implementation is based on 
the Note.pdf.

Sample 1:
For the standard system below

    Objective Function
    f(x) = 3x_1 + x_2 + 9x_3 + x_4 -> min
    with constraints
    x_1 + 2x_2 + x_4 = 4
    x_2 + x_3 - x_4 = 2
    x_1, x_2, x_3, x_4 >= 0

the number of variable is 4
the number of constraint is 2

we have a matrix 

S = [3 1 9  1 0  # Coefficients of the Objective Function; Last 0 is for aligned
     1 2 0  1 3  # Coefficients of the first constraint
     0 1 1 -1 2] # Coefficients of the second constraint
'''

n_var        = 4
n_constraint = 2

s = np.array([[3, 1, 9, 1, 0], [1, 0, 2, 1, 4], [0, 1, 1, -1, 2]])

'''
Inital Step-up and check 
'''

if np.shape(s)[0] != n_constraint - 1 or np.shape(s)[1] != n_var - 1:
    print("The Dimension of the matrix is wrong!")
    
f = s[0,:-1]
A = s[1:,:-1]
l = s[1:,-1:]
c = s[0,:-1].transpose()

n = n_var
m = n_constraint

Basis = np.zeros(m, dtype=np.int64)
Rest = np.zeros((n-m), dtype=np.int64)

for i in range(m):
    Basis[i] = i
for i in range(n-m):
    Rest[i] = m+i

while 1 == 1:
    
    B = np.zeros((m, m), dtype=np.int64)
    N = np.zeros((m, n-m), dtype=np.int64)
    
    for i in range(m):
        B[:,i] = A[:,Basis[i]]
    for i in range(n - m):
        N[:,i] = A[:,Rest[i]]
    
    x_B = inv(B).dot(l)
    
    x = np.zeros((n, 1))
    
    for i in range(m):
        x[Basis[i]] = x_B[i]
    
    fx = f.dot(x)
    
    C_B = np.zeros((m), dtype=np.int64)
    C_N = np.zeros((n-m), dtype=np.int64)
    
    for i in range(m):
        C_B[i] = c[Basis[i]]
        
    for i in range(n-m):
        C_N[i] = c[Rest[i]]
    
    r = C_N-C_B.dot(inv(B)).dot(N)
    
    print("B\n", B, "\nN\n", N, "\nC_B\n", C_B, "\nC_N\n", C_N, "\nr\n",r, "\nx\n", x, "\nfx\n",fx)
    
    if min(r) >= 0:
        print("We are done and the result is", fx[0])
        break
    if max(r) < 0:
        print("There is no solution")
        break
        
    entering = np.argmin(r)+m
    
    w = inv(B).dot(A[:,entering])
    
    leaving = np.where(np.transpose(x_B)/w > 0, np.transpose(x_B)/w, np.inf).argmin()
    
    for i in range(m):
        if Basis[i] == leaving:
            Basis[i] = entering;
            break
        
    for i in range(n-m):
        if Rest[i] == entering:
            Rest[i] = leaving;
            break
    
    Basis.sort()
    Rest.sort()
    print("w\n",w,"\nNew Bases\n", Basis)
    print("------------------------------------------")
    
    
'''
Few more samples
#Q4
n_var        = 5
n_constraint = 3

s = np.array([[3,-1,-4,0,0,0],[0,-1,1,1,0,1],[-5,1,1,0,0,2],[-8,1,2,0,-1,3]])

#Q5
n_var        = 6
n_constraint = 3
s = np.array([[-2,-3,0,1,0,0,0],[2,-1,0,-2,1,0,16],[3,2,1,-3,0,0,18],[-1,3,0,4,0,1,24]])

# in class samplee
n_var        = 5
n_constraint = 3
s = np.array([[4,6,8,0,0,0],[-1,1,1,0,0,11],[1,1,0,1,0,27],[2,5,0,0,1,90]])
'''