from __future__ import annotations
import collections
import configparser
import copy
import os
import os.path
import re
from typing import (
from coverage.exceptions import ConfigError
from coverage.misc import isolate_module, human_sorted_items, substitute_variables
from coverage.tomlconfig import TomlConfigParser, TomlDecodeError
from coverage.types import (
def read_coverage_config(config_file: bool | str, warn: Callable[[str], None], **kwargs: TConfigValueIn) -> CoverageConfig:
    """Read the coverage.py configuration.

    Arguments:
        config_file: a boolean or string, see the `Coverage` class for the
            tricky details.
        warn: a function to issue warnings.
        all others: keyword arguments from the `Coverage` class, used for
            setting values in the configuration.

    Returns:
        config:
            config is a CoverageConfig object read from the appropriate
            configuration file.

    """
    config = CoverageConfig()
    if config_file:
        files_to_try = config_files_to_try(config_file)
        for fname, our_file, specified_file in files_to_try:
            config_read = config.from_file(fname, warn, our_file=our_file)
            if config_read:
                break
            if specified_file:
                raise ConfigError(f"Couldn't read {fname!r} as a config file")
    env_data_file = os.getenv('COVERAGE_FILE')
    if env_data_file:
        config.data_file = env_data_file
    debugs = os.getenv('COVERAGE_DEBUG')
    if debugs:
        config.debug.extend((d.strip() for d in debugs.split(',')))
    config.from_args(**kwargs)
    config.post_process()
    return config