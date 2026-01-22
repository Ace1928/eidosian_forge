import colorsys
import contextlib
import dataclasses
import functools
import gzip
import importlib
import importlib.util
import itertools
import json
import logging
import math
import numbers
import os
import platform
import queue
import random
import re
import secrets
import shlex
import socket
import string
import sys
import tarfile
import tempfile
import threading
import time
import types
import urllib
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from importlib import import_module
from sys import getsizeof
from types import ModuleType
from typing import (
import requests
import yaml
import wandb
import wandb.env
from wandb.errors import AuthenticationError, CommError, UsageError, term
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, runid
from wandb.sdk.lib.json_util import dump, dumps
from wandb.sdk.lib.paths import FilePathStr, StrPath
def ensure_matplotlib_figure(obj: Any) -> Any:
    """Extract the current figure from a matplotlib object.

    Return the object itself if it's a figure.
    Raises ValueError if the object can't be converted.
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.spines import Spine

    def is_frame_like(self: Any) -> bool:
        """Return True if directly on axes frame.

        This is useful for determining if a spine is the edge of an
        old style MPL plot. If so, this function will return True.
        """
        position = self._position or ('outward', 0.0)
        if isinstance(position, str):
            if position == 'center':
                position = ('axes', 0.5)
            elif position == 'zero':
                position = ('data', 0)
        if len(position) != 2:
            raise ValueError('position should be 2-tuple')
        position_type, amount = position
        if position_type == 'outward' and amount == 0:
            return True
        else:
            return False
    Spine.is_frame_like = is_frame_like
    if obj == matplotlib.pyplot:
        obj = obj.gcf()
    elif not isinstance(obj, Figure):
        if hasattr(obj, 'figure'):
            obj = obj.figure
            if not isinstance(obj, Figure):
                raise ValueError('Only matplotlib.pyplot or matplotlib.pyplot.Figure objects are accepted.')
    return obj