import warnings
import numpy as np
import pandas as pd
from scipy import stats
def impute_ros(observations, censorship, df=None, min_uncensored=2, max_fraction_censored=0.8, substitution_fraction=0.5, transform_in=np.log, transform_out=np.exp, as_array=True):
    """
    Impute censored dataset using Regression on Order Statistics (ROS).

    Method described in *Nondetects and Data Analysis* by Dennis R.
    Helsel (John Wiley, 2005) to estimate the left-censored (non-detect)
    values of a dataset. When there is insufficient non-censorded data,
    simple substitution is used.

    Parameters
    ----------
    observations : str or array-like
        Label of the column or the float array of censored observations

    censorship : str
        Label of the column or the bool array of the censorship
        status of the observations.

          * True if censored,
          * False if uncensored

    df : DataFrame, optional
        If `observations` and `censorship` are labels, this is the
        DataFrame that contains those columns.

    min_uncensored : int (default is 2)
        The minimum number of uncensored values required before ROS
        can be used to impute the censored observations. When this
        criterion is not met, simple substituion is used instead.

    max_fraction_censored : float (default is 0.8)
        The maximum fraction of censored data below which ROS can be
        used to impute the censored observations. When this fraction is
        exceeded, simple substituion is used instead.

    substitution_fraction : float (default is 0.5)
        The fraction of the detection limit to be used during simple
        substitution of the censored values.

    transform_in : callable (default is np.log)
        Transformation to be applied to the values prior to fitting a
        line to the plotting positions vs. uncensored values.

    transform_out : callable (default is np.exp)
        Transformation to be applied to the imputed censored values
        estimated from the previously computed best-fit line.

    as_array : bool (default is True)
        When True, a numpy array of the imputed observations is
        returned. Otherwise, a modified copy of the original dataframe
        with all of the intermediate calculations is returned.

    Returns
    -------
    imputed : {ndarray, DataFrame}
        The final observations where the censored values have either been
        imputed through ROS or substituted as a fraction of the
        detection limit.

    Notes
    -----
    This function requires pandas 0.14 or more recent.
    """
    if df is None:
        df = pd.DataFrame({'obs': observations, 'cen': censorship})
        observations = 'obs'
        censorship = 'cen'
    N_observations = df.shape[0]
    N_censored = df[censorship].astype(int).sum()
    N_uncensored = N_observations - N_censored
    fraction_censored = N_censored / N_observations
    if N_censored == 0:
        output = df[[observations, censorship]].copy()
        output.loc[:, 'final'] = df[observations]
    elif N_uncensored < min_uncensored or fraction_censored > max_fraction_censored:
        output = df[[observations, censorship]].copy()
        output.loc[:, 'final'] = df[observations]
        output.loc[df[censorship], 'final'] *= substitution_fraction
    else:
        output = _do_ros(df, observations, censorship, transform_in, transform_out)
    if as_array:
        output = output['final'].values
    return output