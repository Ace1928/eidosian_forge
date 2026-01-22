from __future__ import annotations
import datetime as dt
import uuid
from functools import partial
from types import FunctionType, MethodType
from typing import (
import numpy as np
import param
from bokeh.model import Model
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from bokeh.models.widgets.tables import (
from bokeh.util.serialization import convert_datetime_array
from pyviz_comms import JupyterComm
from ..depends import transform_reference
from ..io.resources import CDN_DIST, CSS_URLS
from ..io.state import state
from ..reactive import Reactive, ReactiveData
from ..util import (
from ..util.warnings import warn
from .base import Widget
from .button import Button
from .input import TextInput
def download_menu(self, text_kwargs={}, button_kwargs={}):
    """
        Returns a menu containing a TextInput and Button widget to set
        the filename and trigger a client-side download of the data.

        Arguments
        ---------
        text_kwargs: dict
            Keyword arguments passed to the TextInput constructor
        button_kwargs: dict
            Keyword arguments passed to the Button constructor

        Returns
        -------
        filename: TextInput
            The TextInput widget setting a filename.
        button: Button
            The Button that triggers a download.
        """
    text_kwargs = dict(text_kwargs)
    if 'name' not in text_kwargs:
        text_kwargs['name'] = 'Filename'
    if 'value' not in text_kwargs:
        text_kwargs['value'] = 'table.csv'
    filename = TextInput(**text_kwargs)
    button_kwargs = dict(button_kwargs)
    if 'name' not in button_kwargs:
        button_kwargs['name'] = 'Download'
    button = Button(**button_kwargs)
    button.js_on_click({'table': self, 'filename': filename}, code='\n        table.filename = filename.value\n        table.download = !table.download\n        ')
    return (filename, button)