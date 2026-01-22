import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
def get_config_provider():
    """
    Returns the current DatabricksConfigProvider.
    If None, the DefaultConfigProvider will be used.
    """
    global _config_provider
    return _config_provider