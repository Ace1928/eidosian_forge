import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils
def simultaneous_cb(self, alpha=0.05, method='hw', transform='log'):
    """
        Returns a simultaneous confidence band for the survival function.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the desired simultaneous coverage
            probability for the confidence region.  Currently alpha
            must be set to 0.05, giving 95% simultaneous intervals.
        method : str
            The method used to produce the simultaneous confidence
            band.  Only the Hall-Wellner (hw) method is currently
            implemented.
        transform : str
            The used to produce the interval (note that the returned
            interval is on the survival probability scale regardless
            of which transform is used).  Only `log` and `arcsin` are
            implemented.

        Returns
        -------
        lcb : array_like
            The lower confidence limits corresponding to the points
            in `surv_times`.
        ucb : array_like
            The upper confidence limits corresponding to the points
            in `surv_times`.
        """
    method = method.lower()
    if method != 'hw':
        msg = 'only the Hall-Wellner (hw) method is implemented'
        raise ValueError(msg)
    if alpha != 0.05:
        raise ValueError('alpha must be set to 0.05')
    transform = transform.lower()
    s2 = self.surv_prob_se ** 2 / self.surv_prob ** 2
    nn = self.n_risk
    if transform == 'log':
        denom = np.sqrt(nn) * np.log(self.surv_prob)
        theta = 1.3581 * (1 + nn * s2) / denom
        theta = np.exp(theta)
        lcb = self.surv_prob ** (1 / theta)
        ucb = self.surv_prob ** theta
    elif transform == 'arcsin':
        k = 1.3581
        k *= (1 + nn * s2) / (2 * np.sqrt(nn))
        k *= np.sqrt(self.surv_prob / (1 - self.surv_prob))
        f = np.arcsin(np.sqrt(self.surv_prob))
        v = np.clip(f - k, 0, np.inf)
        lcb = np.sin(v) ** 2
        v = np.clip(f + k, -np.inf, np.pi / 2)
        ucb = np.sin(v) ** 2
    else:
        raise ValueError('Unknown transform')
    return (lcb, ucb)