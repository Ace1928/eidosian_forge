import numpy as np
from scipy import stats
import pandas as pd
def f_pred(p):
    """Prediction function as function of params
        """
    pred = self.model.predict(p, exog, which=which, **pred_kwds)
    if average:
        pred = (pred.T * agg_weights.T).mean(-1).T
    return pred