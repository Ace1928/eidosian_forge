from __future__ import annotations
import functools
import importlib
import json
import logging
import mimetypes
import os
import pathlib
import re
import textwrap
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import (
import param
from bokeh.embed.bundle import (
from bokeh.model import Model
from bokeh.models import ImportedStyleSheet
from bokeh.resources import Resources as BkResources, _get_server_urls
from bokeh.settings import settings as _settings
from jinja2.environment import Environment
from jinja2.loaders import FileSystemLoader
from markupsafe import Markup
from ..config import config, panel_extension as extension
from ..util import isurl, url_path
from .state import state
def patch_model_css(root, dist_url):
    """
    Temporary patch for Model.css property used by Panel to provide
    stylesheets for components.

    ALERT: Should find better solution before official Bokeh 3.x compatible release
    """
    doc = root.document
    if doc:
        held = doc.callbacks.hold_value
        events = list(doc.callbacks._held_events)
        doc.hold()
    for stylesheet in root.select({'type': ImportedStyleSheet}):
        patch_stylesheet(stylesheet, dist_url)
    if doc:
        doc.callbacks._held_events = events
        if held:
            doc.callbacks._hold = held
        else:
            doc.unhold()