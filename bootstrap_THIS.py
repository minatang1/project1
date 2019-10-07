def bootstrap(X, z, statistic, method, number_of_bootstraps, la):
    stat = np.empty([number_of_bootstraps, 21])
    max_idx = X.shape[0] # highest number of indices
    beta = np.empty([number_of_bootstraps, 21])
    z_predict = np.empty([number_of_bootstraps, X.shape[0]])
    R2_lasso = np.zeros(number_of_bootstraps)
    
    np.random.seed(4155)
    
    for i in range(number_of_bootstraps):
           
        if method == 'OLS':
            idx = np.random.randint(0, max_idx, max_idx)
            beta[i], z_predict[i] = linreg_ols(X[idx], z[idx])
                            
        elif method == 'Ridge':
            idx = np.random.randint(0, max_idx, max_idx)
            beta[i], z_predict[i] = ridgereg(X[idx], z[idx], la)
            
        elif method == 'LASSO':
            idx = np.random.randint(0, max_idx, max_idx)
            beta[i], z_predict[i], R2_lasso[i] = lassoreg(X[idx], z[idx], la)
            
        else:
            break
         
        if statistic:
            stat[i] = statistic(beta)
        
    if not statistic:
        if method == 'OLS' or method == 'Ridge':
            return beta, z_predict
        if method == 'LASSO':
            return beta, z_predict, R2_lasso