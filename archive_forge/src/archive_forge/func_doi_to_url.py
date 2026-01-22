import os
import sys
import ftplib
import warnings
from .utils import parse_url
def doi_to_url(doi):
    """
    Follow a DOI link to resolve the URL of the archive.

    Parameters
    ----------
    doi : str
        The DOI of the archive.

    Returns
    -------
    url : str
        The URL of the archive in the data repository.

    """
    import requests
    response = requests.get(f'https://doi.org/{doi}', timeout=5)
    url = response.url
    if 400 <= response.status_code < 600:
        raise ValueError(f'Archive with doi:{doi} not found (see {url}). Is the DOI correct?')
    return url