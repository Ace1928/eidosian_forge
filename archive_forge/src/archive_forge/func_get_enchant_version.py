import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def get_enchant_version():
    """Get the version string for the underlying enchant library."""
    return _e.get_version().decode()