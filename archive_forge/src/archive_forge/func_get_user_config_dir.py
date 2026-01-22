import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def get_user_config_dir():
    """Return the path that will be used by some
    Enchant providers to look for custom dictionaries.
    """
    return _e.get_user_config_dir().decode()