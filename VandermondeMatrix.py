def VandermondeMatrix(x, y, n):
    X = np.c_[np.ones(len(x))]
    for i in range(1, n+1):
        # x-terms
        X = np.c_[X, x**(i)]
        # y-terms
        X = np.c_[X, y**(i)]
        # Cross terms
        for j in range(i-1, 0, -1):
            X = np.c_[X, (x**(j))*(y**(i-j))]
            
    return X