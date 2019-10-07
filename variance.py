# Variance
def var(y_data, y_model):
    n = 20
    return np.sum((y_model - np.mean(y_model))**2) / n