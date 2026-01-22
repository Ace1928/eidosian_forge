import typing
import urllib.parse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import List, Optional, Union
from requests import get
from pennylane.data.base import Dataset
from pennylane.data.base.hdf5 import open_hdf5_s3
from .foldermap import DataPath, FolderMapView, ParamArg
from .params import DEFAULT, FULL, format_params
def load_interactive():
    """Download a dataset using an interactive load prompt.

    Returns:
        :class:`~pennylane.data.Dataset`

    **Example**

    .. seealso:: :func:`~.load`, :func:`~.list_attributes`, :func:`~.list_datasets`.

    .. code-block :: pycon

        >>> qml.data.load_interactive()
        Please select a data name:
            1) qspin
            2) qchem
        Choice [1-2]: 1
        Please select a sysname:
            ...
        Please select a periodicity:
            ...
        Please select a lattice:
            ...
        Please select a layout:
            ...
        Please select attributes:
            ...
        Force download files? (Default is no) [y/N]: N
        Folder to download to? (Default is pwd, will download to /datasets subdirectory):

        Please confirm your choices:
        dataset: qspin/Ising/open/rectangular/4x4
        attributes: ['parameters', 'ground_states']
        force: False
        dest folder: /Users/jovyan/Downloads/datasets
        Would you like to continue? (Default is yes) [Y/n]:
        <Dataset = description: qspin/Ising/open/rectangular/4x4, attributes: ['parameters', 'ground_states']>
    """
    foldermap = _get_foldermap()
    data_struct = _get_data_struct()
    node = foldermap
    data_name = _interactive_request_single(node, 'data name')
    description = {}
    value = data_name
    params = data_struct[data_name]['params']
    for param in params:
        node = node[value]
        value = _interactive_request_single(node, param)
        description[param] = value
    attributes = _interactive_request_attributes([attribute for attribute in data_struct[data_name]['attributes'] if attribute not in params])
    force = input('Force download files? (Default is no) [y/N]: ') in ['y', 'Y']
    dest_folder = Path(input('Folder to download to? (Default is pwd, will download to /datasets subdirectory): '))
    print('\nPlease confirm your choices:')
    print('dataset:', '/'.join([data_name] + [description[param] for param in params]))
    print('attributes:', attributes)
    print('force:', force)
    print('dest folder:', dest_folder / 'datasets')
    approve = input('Would you like to continue? (Default is yes) [Y/n]: ')
    if approve not in ['Y', '', 'y']:
        print('Aborting and not downloading!')
        return None
    return load(data_name, attributes=attributes, folder_path=dest_folder, force=force, **description)[0]