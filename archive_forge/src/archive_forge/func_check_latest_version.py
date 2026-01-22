import os
from looseversion import LooseVersion
from .info import URL as __url__, STATUS as __status__, __version__
from .utils.config import NipypeConfig
from .utils.logger import Logging
from .refs import due
from .pkg_info import get_pkg_info as _get_pkg_info
from .pipeline import Node, MapNode, JoinNode, Workflow
from .interfaces import (
def check_latest_version(raise_exception=False):
    """
    Check for the latest version of the library.

    Parameters
    ----------
    raise_exception: bool
        Raise a RuntimeError if a bad version is being used
    """
    import etelemetry
    logger = logging.getLogger('nipype.utils')
    return etelemetry.check_available_version('nipy/nipype', __version__, logger, raise_exception)