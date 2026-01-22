from statsmodels.compat.python import lrange
from io import StringIO
from os import environ, makedirs
from os.path import abspath, dirname, exists, expanduser, join
import shutil
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen
import numpy as np
from pandas import Index, read_csv, read_stata
def webuse(data, baseurl='https://www.stata-press.com/data/r11/', as_df=True):
    """
    Download and return an example dataset from Stata.

    Parameters
    ----------
    data : str
        Name of dataset to fetch.
    baseurl : str
        The base URL to the stata datasets.
    as_df : bool
        Deprecated. Always returns a DataFrame

    Returns
    -------
    dta : DataFrame
        A DataFrame containing the Stata dataset.

    Examples
    --------
    >>> dta = webuse('auto')

    Notes
    -----
    Make sure baseurl has trailing forward slash. Does not do any
    error checking in response URLs.
    """
    url = urljoin(baseurl, data + '.dta')
    return read_stata(url)