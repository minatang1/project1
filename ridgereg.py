def ridgereg(X, z, lambda_ridge):

    # Solving for beta
    beta_ridge = np.linalg.inv(np.transpose(X).dot(X) + lambda_ridge*np.identity(X.shape[1])).dot(np.transpose(X)).dot(z)

    y_predict_ridge = X @ beta_ridge
    
    return beta_ridge, y_predict_ridge