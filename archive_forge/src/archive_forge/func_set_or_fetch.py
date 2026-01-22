import configparser
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from project import ProjectData
from .commands import run_qmlimportscanner
from . import DEFAULT_APP_ICON
def set_or_fetch(self, config_property_val, config_property_key, config_property_group='app'):
    """
        Write to config_file if 'config_property_key' is known without config_file
        Fetch and return from config_file if 'config_property_key' is unknown, but
        config_file exists
        Otherwise, raise an exception
        """
    if config_property_val:
        self.set_value(config_property_group, config_property_key, str(config_property_val))
        return config_property_val
    elif self.get_value(config_property_group, config_property_key):
        return self.get_value(config_property_group, config_property_key)
    else:
        raise RuntimeError(f'[DEPLOY] No {config_property_key} specified in config file or as cli option')