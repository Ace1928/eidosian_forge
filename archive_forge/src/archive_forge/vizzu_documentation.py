from __future__ import annotations
import datetime as dt
import sys
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..reactive import SyncableData
from ..util import isdatetime, lazy_load
from .base import ModelPane

        Register a callback to be executed when any element in the
        chart is clicked on.

        Arguments
        ---------
        callback: (callable)
            The callback to run on click events.
        