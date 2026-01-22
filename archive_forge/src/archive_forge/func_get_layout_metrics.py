from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def get_layout_metrics() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[LayoutViewport, VisualViewport, dom.Rect, LayoutViewport, VisualViewport, dom.Rect]]:
    """
    Returns metrics relating to the layouting of the page, such as viewport bounds/scale.

    :returns: A tuple with the following items:

        0. **layoutViewport** - Deprecated metrics relating to the layout viewport. Is in device pixels. Use ``cssLayoutViewport`` instead.
        1. **visualViewport** - Deprecated metrics relating to the visual viewport. Is in device pixels. Use ``cssVisualViewport`` instead.
        2. **contentSize** - Deprecated size of scrollable area. Is in DP. Use ``cssContentSize`` instead.
        3. **cssLayoutViewport** - Metrics relating to the layout viewport in CSS pixels.
        4. **cssVisualViewport** - Metrics relating to the visual viewport in CSS pixels.
        5. **cssContentSize** - Size of scrollable area in CSS pixels.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.getLayoutMetrics'}
    json = (yield cmd_dict)
    return (LayoutViewport.from_json(json['layoutViewport']), VisualViewport.from_json(json['visualViewport']), dom.Rect.from_json(json['contentSize']), LayoutViewport.from_json(json['cssLayoutViewport']), VisualViewport.from_json(json['cssVisualViewport']), dom.Rect.from_json(json['cssContentSize']))