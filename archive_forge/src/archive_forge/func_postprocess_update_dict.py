from __future__ import annotations
import copy
import hashlib
import inspect
import json
import os
import random
import secrets
import string
import sys
import threading
import time
import warnings
import webbrowser
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Literal, Sequence, cast
from urllib.parse import urlparse, urlunparse
import anyio
import fastapi
import httpx
from anyio import CapacityLimiter
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import (
from gradio.blocks_events import BlocksEvents, BlocksMeta
from gradio.context import Context
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.events import (
from gradio.exceptions import (
from gradio.helpers import create_tracker, skip, special_args
from gradio.state_holder import SessionState
from gradio.themes import Default as DefaultTheme
from gradio.themes import ThemeClass as Theme
from gradio.tunneling import (
from gradio.utils import (
def postprocess_update_dict(block: Component | BlockContext, update_dict: dict, postprocess: bool=True):
    """
    Converts a dictionary of updates into a format that can be sent to the frontend to update the component.
    E.g. {"value": "2", "visible": True, "invalid_arg": "hello"}
    Into -> {"__type__": "update", "value": 2.0, "visible": True}
    Parameters:
        block: The Block that is being updated with this update dictionary.
        update_dict: The original update dictionary
        postprocess: Whether to postprocess the "value" key of the update dictionary.
    """
    value = update_dict.pop('value', components._Keywords.NO_VALUE)
    update_dict = {k: getattr(block, k) for k in update_dict if hasattr(block, k)}
    if value is not components._Keywords.NO_VALUE:
        if postprocess:
            update_dict['value'] = block.postprocess(value)
            if isinstance(update_dict['value'], (GradioModel, GradioRootModel)):
                update_dict['value'] = update_dict['value'].model_dump()
        else:
            update_dict['value'] = value
    update_dict['__type__'] = 'update'
    return update_dict