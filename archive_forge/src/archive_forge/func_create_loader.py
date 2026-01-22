import logging
import os
from botocore import BOTOCORE_ROOT
from botocore.compat import HAS_GZIP, OrderedDict, json
from botocore.exceptions import DataNotFoundError, UnknownServiceError
from botocore.utils import deep_merge
def create_loader(search_path_string=None):
    """Create a Loader class.

    This factory function creates a loader given a search string path.

    :type search_string_path: str
    :param search_string_path: The AWS_DATA_PATH value.  A string
        of data path values separated by the ``os.path.pathsep`` value,
        which is typically ``:`` on POSIX platforms and ``;`` on
        windows.

    :return: A ``Loader`` instance.

    """
    if search_path_string is None:
        return Loader()
    paths = []
    extra_paths = search_path_string.split(os.pathsep)
    for path in extra_paths:
        path = os.path.expanduser(os.path.expandvars(path))
        paths.append(path)
    return Loader(extra_search_paths=paths)