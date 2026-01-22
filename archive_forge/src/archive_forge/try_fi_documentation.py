import numpy as np
from scipy.special import gamma
from scipy import signal
from statsmodels.tsa.arima_process import (lpol_fiar, lpol_fima,

    array([-0.029952  , -0.01100641, -0.00410998, -0.00299859])
    >>> d=0.4; j=np.arange(1000);ri=gamma(d+j)/(gamma(j+1)*gamma(d))
    >>> # (1-L)^d, d<1 is
    >>> lfilter([1], ri, [1]+[0]*30)
    array([ 1.        , -0.4       , -0.12      , -0.064     , -0.0416    ,
          -0.029952  , -0.0229632 , -0.01837056, -0.01515571, -0.01279816,
          -0.01100641, -0.0096056 , -0.00848495, -0.00757118, -0.00681406,
          -0.00617808, -0.0056375 , -0.00517324, -0.00477087, -0.00441934,
          -0.00410998, -0.00383598, -0.00359188, -0.00337324, -0.00317647,
          -0.00299859, -0.00283712, -0.00269001, -0.00255551, -0.00243214,
          -0.00231864])
    >>> # verified for points [[5,10,20,25]] at 4 decimals with Bhardwaj, Swanson, Journal of Eonometrics 2006
    