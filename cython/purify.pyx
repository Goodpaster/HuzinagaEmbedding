#A method to purify reduced density matrices
#By DanG

import numpy as np

max_count = 5000
#McWeeny's original purification method
def McWeeny_pur(dmat):
    print (dmat)
    print (np.linalg.eig(dmat))
    dmat = np.multiply(0.5, dmat)
    count = 0
    fun_zero = 1e-15
    idem_diff = 1.
    while(idem_diff > fun_zero and count < max_count):
        if (count % 1 == 0):
            print (' iter: {0:>3d}       |idem_diff|: {1:12.12f}     '.format((count+1), idem_diff))
        dmat = np.subtract(np.multiply(3, np.linalg.matrix_power(dmat, 2)),
                           np.multiply(2, np.linalg.matrix_power(dmat, 3)))
        count += 1
        idem_diff = (abs(np.sum(np.subtract(dmat, np.linalg.matrix_power(dmat, 2)))))

    if(count >= max_count and idem_diff > fun_zero): raise StopIteration('Purification Failure!', idem_diff)
    print ('Density Purified!') 
    print (np.multiply(2, dmat))
    print (np.linalg.eig((np.multiply(2, dmat))))
    return np.multiply(2, dmat)
    

#Truflandier and Dianzinga's purification method
def Truf_pur(dmat):
    dmat = np.multiply(0.5, dmat)
    count = 0
    fun_zero = 1e-12
    idem_diff = 1.
    while(idem_diff > fun_zero and count < max_count):
        if (count % 1 == 0):
            print (' iter: {0:>3d}       |idem_diff|: {1:12.12f}     '.format((count+1), idem_diff))
        c_top = np.trace(np.subtract(np.linalg.matrix_power(dmat, 2),
                         np.linalg.matrix_power(dmat, 3)))
        c_bot = np.trace(np.subtract(dmat, np.linalg.matrix_power(dmat, 2)))
        c = np.divide(c_top, c_bot)

        gamma = (1./3.) - (2./3.) * c

        del_omega = 2. * (np.subtract(2 * np.linalg.matrix_power(dmat, 3),
                          (3 * np.linalg.matrix_power(dmat, 2))) + dmat)
        del_lagrange = np.subtract(del_omega, (6 * gamma * np.subtract(dmat,
                       np.linalg.matrix_power(dmat, 2))))

        dmat = np.subtract(dmat, (0.5 * del_lagrange))

        count += 1
        idem_diff = (abs(np.sum(np.subtract(dmat, np.linalg.matrix_power(dmat, 2)))))

    if(count >= max_count and idem_diff > fun_zero): raise StopIteration('Purification Failure!', idem_diff)
    print ('Density Purified!') 
    return np.multiply(2, dmat)

#Palser and Manolopoulos purification method
def Palser_pur(dmat):
    dmat = np.multiply(0.5, dmat)
    count = 0
    fun_zero = 1e-12
    idem_diff = 1.
    while(idem_diff > fun_zero and count < max_count):
        if (count % 1000 == 0):
            print (' iter: {0:>3d}       |idem_diff|: {1:12.12f}     '.format((count+1), idem_diff))
        c_top = np.trace(np.subtract(np.linalg.matrix_power(dmat, 2),
                         np.linalg.matrix_power(dmat, 3)))
        c_bot = np.trace(np.subtract(dmat, np.linalg.matrix_power(dmat, 2)))
        c = np.divide(c_top, c_bot)
        if c < 0.5:
            dmat_top = np.multiply((1 - 2*c), dmat) + np.multiply((1 + c),
                       np.subtract(np.linalg.matrix_power(dmat, 2), 
                       np.linalg.matrix_power(dmat, 3)))

            dmat = np.multiply((1./(1 - c)), dmat_top)

        else:
            dmat_top = np.multiply((1 + c),
                       np.subtract(np.linalg.matrix_power(dmat, 2), 
                       np.linalg.matrix_power(dmat, 3)))

            dmat = np.multiply((1./c), dmat_top)

        count += 1
        idem_diff = (abs(np.sum(np.subtract(dmat, np.linalg.matrix_power(dmat, 2)))))

    if(count >= max_count and idem_diff > fun_zero): raise StopIteration('Purification Failure!', idem_diff)
    print ('Density Purified!') 
    return np.multiply(2, dmat)

#Convert MO density matrix to natural orbitals
def Nat_orb_mat(dmat, nelec):
    eig_val, eig_mat = np.linalg.eig(dmat)
    print (eig_mat)
    eig_vals = np.array([eig_val]).T
    print (eig_vals)
    return np.diag(eig_val)
    #return (eig_mat * eig_vals)
    
