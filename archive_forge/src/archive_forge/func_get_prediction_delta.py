import numpy as np
from scipy import stats
import pandas as pd
def get_prediction_delta(self, exog=None, which='mean', average=False, agg_weights=None, transform=True, row_labels=None, pred_kwds=None):
    """
    compute prediction results

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    which : str
        The statistic that is prediction. Which statistics are available
        depends on the model.predict method.
    average : bool
        If average is True, then the mean prediction is computed, that is,
        predictions are computed for individual exog and then them mean over
        observation is used.
        If average is False, then the results are the predictions for all
        observations, i.e. same length as ``exog``.
    agg_weights : ndarray, optional
        Aggregation weights, only used if average is True.
        The weights are not normalized.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    pred_kwargs :
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models.
        See the predict method of the model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """
    exog, row_labels = _get_exog_predict(self, exog=exog, transform=transform, row_labels=row_labels)
    if agg_weights is None:
        agg_weights = np.array(1.0)

    def f_pred(p):
        """Prediction function as function of params
        """
        pred = self.model.predict(p, exog, which=which, **pred_kwds)
        if average:
            pred = (pred.T * agg_weights.T).mean(-1).T
        return pred
    nlpm = self._get_wald_nonlinear(f_pred)
    res = PredictionResultsDelta(nlpm)
    return res