import numpy as np

np.random.seed(0)

pool = 200000000 * [0] + 100000000 * [1]
t = 0.01
eps = 0.01
hoeffding_amount = np.ceil(np.log(2 / eps) / (2 * t ** 2))
subsample = np.random.choice(pool, size=hoeffding_amount, replace=True)
print(subsample.mean())
