def confidence_interval(X, beta, z_level, std):
    
    XtXi = np.linalg.inv(np.transpose(X).dot(X))

    num_beta = beta.shape[0]
    
    diagonals = np.zeros(num_beta)
    
    for j in range(num_beta):
        diagonals[j] = np.sqrt(XtXi[j][j])
        
    cint_upper = np.zeros(num_beta)
    cint_lower = np.zeros(num_beta)
    
    for i in range(num_beta):
        
        cint_upper[i] = beta[i] + z_level*std
        cint_lower[i] = beta[i] - z_level*std
        
    return cint_upper, cint_lower, diagonals