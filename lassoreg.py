from sklearn.linear_model import Lasso

def lassoreg(X, z, lambda_lasso):
    
    las = Lasso(alpha=lambda_lasso)
    las.fit(X, z)
    
    beta = las.coef_[:,np.newaxis]
    
    y_predict_lasso = X @ beta
    
    R2_lasso = las.score(X, z)
    
    return beta, y_predict_lasso, R2_lasso