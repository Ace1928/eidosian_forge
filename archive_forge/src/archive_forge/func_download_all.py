import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def download_all(directory=None):
    """Download all datasets for use with scikit-image offline.

    Scikit-image datasets are no longer shipped with the library by default.
    This allows us to use higher quality datasets, while keeping the
    library download size small.

    This function requires the installation of an optional dependency, pooch,
    to download the full dataset. Follow installation instruction found at

        https://scikit-image.org/docs/stable/user_guide/install.html

    Call this function to download all sample images making them available
    offline on your machine.

    Parameters
    ----------
    directory: path-like, optional
        The directory where the dataset should be stored.

    Raises
    ------
    ModuleNotFoundError:
        If pooch is not install, this error will be raised.

    Notes
    -----
    scikit-image will only search for images stored in the default directory.
    Only specify the directory if you wish to download the images to your own
    folder for a particular reason. You can access the location of the default
    data directory by inspecting the variable ``skimage.data.data_dir``.
    """
    if _image_fetcher is None:
        raise ModuleNotFoundError('To download all package data, scikit-image needs an optional dependency, pooch.To install pooch, follow our installation instructions found at https://scikit-image.org/docs/stable/user_guide/install.html')
    old_dir = _image_fetcher.path
    try:
        if directory is not None:
            directory = osp.expanduser(directory)
            _image_fetcher.path = directory
        _ensure_cache_dir(target_dir=_image_fetcher.path)
        for data_filename in _image_fetcher.registry:
            file_path = _fetch(data_filename)
            if not file_path.startswith(str(_image_fetcher.path)):
                dest_path = osp.join(_image_fetcher.path, data_filename)
                os.makedirs(osp.dirname(dest_path), exist_ok=True)
                shutil.copy2(file_path, dest_path)
    finally:
        _image_fetcher.path = old_dir