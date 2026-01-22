from __future__ import annotations
import logging # isort:skip
from html import escape
from typing import TYPE_CHECKING, Any
from ..core.json_encoder import serialize_json
from ..core.templates import (
from ..document.document import DEFAULT_TITLE
from ..settings import settings
from ..util.serialization import make_globally_unique_css_safe_id
from .util import RenderItem
from .wrappers import wrap_in_onload, wrap_in_safely, wrap_in_script_tag
def div_for_render_item(item: RenderItem) -> str:
    """ Render an HTML div for a Bokeh render item.

    Args:
        item (RenderItem):
            the item to create a div for

    Returns:
        str

    """
    return PLOT_DIV.render(doc=item, macros=MACROS)