import glob
import os
import subprocess
import sys
import tempfile
from distutils import log
from distutils.errors import DistutilsError
from functools import partial
from . import _reqs
from .wheel import Wheel
from .warnings import SetuptoolsDeprecationWarning
def strip_marker(req):
    """
    Return a new requirement without the environment marker to avoid
    calling pip with something like `babel; extra == "i18n"`, which
    would always be ignored.
    """
    import pkg_resources
    req = pkg_resources.Requirement.parse(str(req))
    req.marker = None
    return req