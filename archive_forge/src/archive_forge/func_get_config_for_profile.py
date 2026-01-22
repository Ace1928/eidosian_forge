import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
def get_config_for_profile(profile):
    """
    [Deprecated] Reads from the filesystem and gets a DatabricksConfig for the
    specified profile. If it does not exist, then return a DatabricksConfig with fields set
    to None.

    Internal callers should prefer get_config() to use user-specified overrides, and
    to return appropriate error messages as opposited to invalid configurations.

    If you want to read from a specific profile, please instead use
    'ProfileConfigProvider(profile).get_config()'.

    This method is maintained for backwards-compatibility. It may be removed in future versions.

    Returns:
        DatabricksConfig
    """
    profile = profile if profile else DEFAULT_SECTION
    config = EnvironmentVariableConfigProvider().get_config()
    if config and config.is_valid:
        return config
    config = ProfileConfigProvider(profile).get_config()
    if config:
        return config
    return DatabricksConfig.empty()