import hashlib
import itertools
import json
import os
import shutil
from collections import namedtuple
from urllib.request import urlretrieve
from ..rcparams import rcParams
from .io_netcdf import from_netcdf
def load_arviz_data(dataset=None, data_home=None, **kwargs):
    """Load a local or remote pre-made dataset.

    Run with no parameters to get a list of all available models.

    The directory to save to can also be set with the environment
    variable `ARVIZ_HOME`. The checksum of the dataset is checked against a
    hardcoded value to watch for data corruption.

    Run `az.clear_data_home` to clear the data directory.

    Parameters
    ----------
    dataset : str
        Name of dataset to load.
    data_home : str, optional
        Where to save remote datasets
    **kwargs : dict, optional
        Keyword arguments passed to :func:`arviz.from_netcdf`.

    Returns
    -------
    xarray.Dataset

    """
    if dataset in LOCAL_DATASETS:
        resource = LOCAL_DATASETS[dataset]
        return from_netcdf(resource.filename, **kwargs)
    elif dataset in REMOTE_DATASETS:
        remote = REMOTE_DATASETS[dataset]
        home_dir = get_data_home(data_home=data_home)
        file_path = os.path.join(home_dir, remote.filename)
        if not os.path.exists(file_path):
            http_type = rcParams['data.http_protocol']
            url = remote.url.replace('http', http_type)
            urlretrieve(url, file_path)
        checksum = _sha256(file_path)
        if remote.checksum != checksum:
            raise IOError(f'{file_path} has an SHA256 checksum ({checksum}) differing from expected ({{remote.checksum}}), file may be corrupted. Run `arviz.clear_data_home()` and try again, or please open an issue.')
        return from_netcdf(file_path, **kwargs)
    elif dataset is None:
        return dict(itertools.chain(LOCAL_DATASETS.items(), REMOTE_DATASETS.items()))
    else:
        raise ValueError('Dataset {} not found! The following are available:\n{}'.format(dataset, list_datasets()))