from __future__ import annotations
import logging # isort:skip
from ...core.properties import Either, Instance, List
from ..dom import DOMNode
from .ui_element import UIElement
 A UI element that can hold other DOM-based UI elements.

    ``Pane`` is a basic building block of DOM-based UIs, and as such it
    doesn't include any properties for controlling its position and other
    visual aspects. These must be configured up by using CSS stylesheets.
    If finer control is needed, use ``Panel`` or ``LayoutDOM`` derived
    models instead.
    