beta_mean_ols = np.mean(beta_bootstrap_ols, axis=0)
beta_mean_ridge = np.mean(beta_bootstrap_ridge, axis=0)
beta_mean_lasso = np.mean(beta_bootstrap_lasso, axis=0)

z_pred_ols = X_test @ beta_mean_ols
z_pred_ridge = X_test @ beta_mean_ridge
z_pred_lasso = X_test @ beta_mean_lasso

print('R2 \n')
print('OLS', R2(z_test, z_pred_ols), '\n RIDGE', R2(z_test, z_pred_ridge), '\n LASSO', R2(z_test, z_pred_lasso))
print('\n MSE \n')
print('OLS', MSE(z_test, z_pred_ols), '\n RIDGE', MSE(z_test, z_pred_ridge), '\n LASSO', MSE(z_test, z_pred_lasso))