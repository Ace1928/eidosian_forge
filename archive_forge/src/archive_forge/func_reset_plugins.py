import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def reset_plugins():
    _clear_plugins()
    _load_preferred_plugins()