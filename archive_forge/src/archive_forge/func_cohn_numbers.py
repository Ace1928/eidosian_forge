import warnings
import numpy as np
import pandas as pd
from scipy import stats
def cohn_numbers(df, observations, censorship):
    """
    Computes the Cohn numbers for the detection limits in the dataset.

    The Cohn Numbers are:

        - :math:`A_j =` the number of uncensored obs above the jth
          threshold.
        - :math:`B_j =` the number of observations (cen & uncen) below
          the jth threshold.
        - :math:`C_j =` the number of censored observations at the jth
          threshold.
        - :math:`\\mathrm{PE}_j =` the probability of exceeding the jth
          threshold
        - :math:`\\mathrm{DL}_j =` the unique, sorted detection limits
        - :math:`\\mathrm{DL}_{j+1} = \\mathrm{DL}_j` shifted down a
          single index (row)

    Parameters
    ----------
    dataframe : DataFrame

    observations : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    -------
    cohn : DataFrame
    """

    def nuncen_above(row):
        """ A, the number of uncensored obs above the given threshold.
        """
        above = df[observations] >= row['lower_dl']
        below = df[observations] < row['upper_dl']
        detect = ~df[censorship]
        return df[above & below & detect].shape[0]

    def nobs_below(row):
        """ B, the number of observations (cen & uncen) below the given
        threshold
        """
        less_than = df[observations] < row['lower_dl']
        less_thanequal = df[observations] <= row['lower_dl']
        uncensored = ~df[censorship]
        censored = df[censorship]
        LTE_censored = df[less_thanequal & censored].shape[0]
        LT_uncensored = df[less_than & uncensored].shape[0]
        return LTE_censored + LT_uncensored

    def ncen_equal(row):
        """ C, the number of censored observations at the given
        threshold.
        """
        censored_index = df[censorship]
        censored_data = df[observations][censored_index]
        censored_below = censored_data == row['lower_dl']
        return censored_below.sum()

    def set_upper_limit(cohn):
        """ Sets the upper_dl DL for each row of the Cohn dataframe. """
        if cohn.shape[0] > 1:
            return cohn['lower_dl'].shift(-1).fillna(value=np.inf)
        else:
            return [np.inf]

    def compute_PE(A, B):
        """ Computes the probability of excedance for each row of the
        Cohn dataframe. """
        N = len(A)
        PE = np.empty(N, dtype='float64')
        PE[-1] = 0.0
        for j in range(N - 2, -1, -1):
            PE[j] = PE[j + 1] + (1 - PE[j + 1]) * A[j] / (A[j] + B[j])
        return PE
    censored_data = df[censorship]
    DLs = pd.unique(df.loc[censored_data, observations])
    DLs.sort()
    if DLs.shape[0] > 0:
        if df[observations].min() < DLs.min():
            DLs = np.hstack([df[observations].min(), DLs])
        cohn = pd.DataFrame(DLs, columns=['lower_dl'])
        cohn.loc[:, 'upper_dl'] = set_upper_limit(cohn)
        cohn.loc[:, 'nuncen_above'] = cohn.apply(nuncen_above, axis=1)
        cohn.loc[:, 'nobs_below'] = cohn.apply(nobs_below, axis=1)
        cohn.loc[:, 'ncen_equal'] = cohn.apply(ncen_equal, axis=1)
        cohn = cohn.reindex(range(DLs.shape[0] + 1))
        cohn.loc[:, 'prob_exceedance'] = compute_PE(cohn['nuncen_above'], cohn['nobs_below'])
    else:
        dl_cols = ['lower_dl', 'upper_dl', 'nuncen_above', 'nobs_below', 'ncen_equal', 'prob_exceedance']
        cohn = pd.DataFrame(np.empty((0, len(dl_cols))), columns=dl_cols)
    return cohn