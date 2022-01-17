import numpy as np

def CorBrownian(VarCovar,mu, sampleSize):

    def Cholesky(X):
        L = np.zeros_like(X)
        n = X.shape[0]

        for i in range(0, n):
            for j in range(0, i+1):
                sum = 0
                for k in range(0, j):
                    sum = sum+ L[i,k]*L[j,k]
                if (i==j):
                    L[i,j] = np.sqrt(X[i,i]-sum)
                else:
                    L[i,j] = 1.0/L[j,j] * (X[i,j]-sum)
        return L

    dim = VarCovar.shape[0]
    Z = np.random.normal(0,1,(sampleSize,dim))
    Y = np.zeros((sampleSize, dim))
    L = Cholesky(VarCovar)

    for iSample in range(sampleSize):
        Y[iSample] =np.transpose(mu) +  L @ np.transpose(Z[iSample])     
    return Y

# Sample n dim normal distributions of sampleSize
sampleSize = 1000
mu = [0,0]
VarCovar = np.matrix('1,0.8; 0.8,1')
print(CorBrownian(VarCovar,mu, sampleSize))
