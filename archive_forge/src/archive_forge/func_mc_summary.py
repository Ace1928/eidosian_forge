import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arma_mle import Arma
def mc_summary(res, rt=None):
    if rt is None:
        rt = np.zeros(res.shape[1])
    nanrows = np.isnan(res).any(1)
    print('fractions of iterations with nans', nanrows.mean())
    res = res[~nanrows]
    print('RMSE')
    print(np.sqrt(((res - rt) ** 2).mean(0)))
    print('mean bias')
    print((res - rt).mean(0))
    print('median bias')
    print(np.median(res - rt, 0))
    print('median bias percent')
    print(np.median((res - rt) / rt * 100, 0))
    print('median absolute error')
    print(np.median(np.abs(res - rt), 0))
    print('positive error fraction')
    print((res > rt).mean(0))