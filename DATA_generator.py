# GENERATING DATA
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

x, y = np.meshgrid(x, y)

z = FrankeFunction(x, y)

x = np.ravel(x)
y = np.ravel(y)
z = np.ravel(z)

z = z + 0.1 * np.random.randn(z.shape[0]) # noise level = 0.1