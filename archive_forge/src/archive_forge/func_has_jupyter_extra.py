import os
import sys
from .json_tools import JSONMixin
from .layer import Layer
from ..io.html import deck_to_html
from ..settings import settings as pydeck_settings
from .view import View
from .view_state import ViewState
from .base_map_provider import BaseMapProvider
from .map_styles import DARK, get_from_map_identifier
def has_jupyter_extra():
    try:
        from ..widget import DeckGLWidget
        DeckGLWidget()
        return True
    except ImportError:
        return False