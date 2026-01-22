from __future__ import annotations
import textwrap
from contextlib import contextmanager
from typing import (
import numpy as np
from bokeh.core.serialization import Serializer
from bokeh.document import Document
from bokeh.document.events import (
from bokeh.document.json import PatchJson
from bokeh.model import DataModel
from bokeh.models import ColumnDataSource, FlexBox, Model
from bokeh.protocol.messages.patch_doc import patch_doc
from .state import state
def patch_cds_msg(model, msg):
    """
    Required for handling messages containing JSON serialized typed
    array from the frontend.
    """
    for event in msg.get('content', {}).get('events', []):
        if event.get('kind') != 'ModelChanged' or event.get('attr') != 'data':
            continue
        cds = model.select_one({'id': event.get('model').get('id')})
        if not isinstance(cds, ColumnDataSource):
            continue
        for col, values in event.get('new', {}).items():
            if isinstance(values, dict):
                event['new'][col] = [v for _, v in sorted(values.items())]