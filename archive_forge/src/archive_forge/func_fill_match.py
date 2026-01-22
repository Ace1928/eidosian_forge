import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging
def fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):
    """
        A utility method that massages the config file and can optionally verify that the values match.

        1. Replace "auto" values with `TrainingArguments` value.

        2. If it wasn't "auto" and `must_match` is true, then check that DS config matches Trainer
        config values and if mismatched add the entry to `self.mismatched` - will assert during
        `trainer_config_finalize` for one or more mismatches.

        """
    config, ds_key = self.find_config_node(ds_key_long)
    if config is None:
        return
    if config.get(ds_key) == 'auto':
        config[ds_key] = hf_val
        return
    if not must_match:
        return
    ds_val = config.get(ds_key)
    if ds_val is not None and ds_val != hf_val:
        self.mismatches.append(f'- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}')