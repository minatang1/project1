# Modified code from last year Piazza
[n, m] = terrain1.shape

patch_size_row = 100
patch_size_col = 50

rows = np.linspace(0,1,patch_size_row)
cols = np.linspace(0,1,patch_size_col)

[C, R] = np.meshgrid(cols, rows)

x = C.reshape(-1,1)
y = R.reshape(-1,1)

num_data = patch_size_row * patch_size_col

num_patches = 10

np.random.seed(3059)

row_starts = np.random.randint(0,n-patch_size_row,num_patches)
col_starts = np.random.randint(0,m-patch_size_col,num_patches)

R2_ols = list()
R2_ridge = list()
MSE_ols = list()
MSE_ridge = list()
#R2_lasso_l = list()

for i,row_start, col_start in zip(np.arange(num_patches), row_starts, col_starts):
    row_end = row_start + patch_size_row
    col_end = col_start + patch_size_col
    
    patch = terrain1[row_start:row_end, col_start:col_end]
    
    z = patch.reshape(-1,1)
    
    X = VandermondeMatrix(x, y, 5)
    
    beta_ols, fitted_patch = linreg_ols(X,z)
    beta_ridge, z_predict_ridge = ridgereg(X, z, lambda_ridge = 0.1)
    beta_lasso, z_predict_lasso, R2_lasso = lassoreg(X, z, lambda_lasso = 0.01)
    
    R2_s = R2(fitted_patch, z)
    MSE_s = MSE(fitted_patch, z)
    
    R2_ols.append(R2_s)
    MSE_ols.append(MSE_s)
    
    R2_r = R2(z_predict_ridge, z)
    MSE_r = MSE(z_predict_ridge, z)
    
    R2_ridge.append(R2_r)
    MSE_ridge.append(MSE_r)
    
    #R2_lasso_l.append(R2_lasso)
    
print("OLS R2 mean", np.mean(R2_ols))
print("RIDGE R2 mean", np.mean(R2_ridge))
#print("LASSO R2 mean", np.mean(R2_lasso_l))
print("OLS MSE mean", np.mean(MSE_ols))
print("RIDGE MSE mean", np.mean(MSE_ridge))

