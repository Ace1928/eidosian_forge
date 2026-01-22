from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import param
from bokeh.models import CustomJS
from ..config import config
from ..reactive import ReactiveHTML
from ..util import classproperty
from .datamodel import _DATA_MODELS, construct_data_model
from .resources import CSS_URLS, bundled_files, get_dist_path
from .state import state

        Generates a layout which allows demoing the component.
        