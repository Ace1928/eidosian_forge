from __future__ import annotations
import inspect
import os
import typing as t
import warnings
def should_warn(key: t.Any) -> bool:
    """Add our own checks for too many deprecation warnings.

    Limit to once per package.
    """
    env_flag = os.environ.get('TRAITLETS_ALL_DEPRECATIONS')
    if env_flag and env_flag != '0':
        return True
    if key not in _deprecations_shown:
        _deprecations_shown.add(key)
        return True
    else:
        return False