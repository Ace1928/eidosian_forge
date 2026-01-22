import contextlib
import logging
import os
from pathlib import Path
import sys
import warnings
import matplotlib as mpl
from matplotlib import _api, _docstring, _rc_params_in_file, rcParamsDefault
def update_nested_dict(main_dict, new_dict):
    """
    Update nested dict (only level of nesting) with new values.

    Unlike `dict.update`, this assumes that the values of the parent dict are
    dicts (or dict-like), so you shouldn't replace the nested dict if it
    already exists. Instead you should update the sub-dict.
    """
    for name, rc_dict in new_dict.items():
        main_dict.setdefault(name, {}).update(rc_dict)
    return main_dict