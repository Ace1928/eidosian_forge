from __future__ import annotations
import configparser
import json
import os
import warnings
from typing import Any
def set_conf_env(conf_dict, envdict=os.environ):
    """Set config values from environment variables

    Looks for variables of the form ``FSSPEC_<protocol>`` and
    ``FSSPEC_<protocol>_<kwarg>``. For ``FSSPEC_<protocol>`` the value is parsed
    as a json dictionary and used to ``update`` the config of the
    corresponding protocol. For ``FSSPEC_<protocol>_<kwarg>`` there is no
    attempt to convert the string value, but the kwarg keys will be lower-cased.

    The ``FSSPEC_<protocol>_<kwarg>`` variables are applied after the
    ``FSSPEC_<protocol>`` ones.

    Parameters
    ----------
    conf_dict : dict(str, dict)
        This dict will be mutated
    envdict : dict-like(str, str)
        Source for the values - usually the real environment
    """
    kwarg_keys = []
    for key in envdict:
        if key.startswith('FSSPEC_') and len(key) > 7 and (key[7] != '_'):
            if key.count('_') > 1:
                kwarg_keys.append(key)
                continue
            try:
                value = json.loads(envdict[key])
            except json.decoder.JSONDecodeError as ex:
                warnings.warn(f'Ignoring environment variable {key} due to a parse failure: {ex}')
            else:
                if isinstance(value, dict):
                    _, proto = key.split('_', 1)
                    conf_dict.setdefault(proto.lower(), {}).update(value)
                else:
                    warnings.warn(f'Ignoring environment variable {key} due to not being a dict: {type(value)}')
        elif key.startswith('FSSPEC'):
            warnings.warn(f'Ignoring environment variable {key} due to having an unexpected name')
    for key in kwarg_keys:
        _, proto, kwarg = key.split('_', 2)
        conf_dict.setdefault(proto.lower(), {})[kwarg.lower()] = envdict[key]