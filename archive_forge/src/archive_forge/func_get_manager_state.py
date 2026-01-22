import os
import sys
import typing
from contextlib import contextmanager
from collections.abc import Iterable
from IPython import get_ipython
from traitlets import (
from json import loads as jsonloads, dumps as jsondumps
from .. import comm
from base64 import standard_b64encode
from .utils import deprecation, _get_frame
from .._version import __protocol_version__, __control_protocol_version__, __jupyter_widgets_base_version__
import inspect
@staticmethod
def get_manager_state(drop_defaults=False, widgets=None):
    """Returns the full state for a widget manager for embedding

        :param drop_defaults: when True, it will not include default value
        :param widgets: list with widgets to include in the state (or all widgets when None)
        :return:
        """
    state = {}
    if widgets is None:
        widgets = _instances.values()
    for widget in widgets:
        state[widget.model_id] = widget._get_embed_state(drop_defaults=drop_defaults)
    return {'version_major': 2, 'version_minor': 0, 'state': state}