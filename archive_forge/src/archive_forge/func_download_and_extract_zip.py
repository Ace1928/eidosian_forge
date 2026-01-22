from .. import utils
from .._lazyload import requests
import os
import tempfile
import urllib.request
import zipfile
def download_and_extract_zip(url, destination):
    """Download a .zip file from a URL and extract it.

    Parameters
    ----------
    url : string
        URL of file to be downloaded
    destination : string
        Directory in which to extract the downloaded zip
    """
    if not os.path.isdir(destination):
        os.mkdir(destination)
    zip_handle = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    download_url(url, zip_handle)
    zip_handle.close()
    unzip(zip_handle.name, destination)