def linreg_ols(X, z):
    
    # Solving for beta
    beta = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(z)
    
    y_predict_ols = X @ beta
    
    return beta, y_predict_ols