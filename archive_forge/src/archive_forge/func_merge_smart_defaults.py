import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def merge_smart_defaults(self, config_store, mode, region_name):
    if mode == 'auto':
        mode = self.resolve_auto_mode(region_name)
    default_configs = self._default_config_resolver.get_default_config_values(mode)
    for config_var in default_configs:
        config_value = default_configs[config_var]
        method = getattr(self, f'_set_{config_var}', None)
        if method:
            method(config_store, config_value)