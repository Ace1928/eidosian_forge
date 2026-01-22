from __future__ import annotations
import operator
import typing
import warnings
from collections.abc import Iterable, Sized
from contextlib import suppress
from typing_extensions import Protocol, runtime_checkable
from urwid import signals
from urwid.canvas import CanvasCombine, SolidCanvas
from .constants import Sizing, VAlign, WHSettings, normalize_valign
from .container import WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, nocache_widget_render_instance

        Return a reversed iterator over the positions in this ListBox.

        If :attr:`body` does not implement :meth:`positions` then iterate
        from above the focus widget up to the top, then from the focus
        widget down to the bottom.  Note that this is not actually the
        reverse of what `__iter__()` produces, but this is the best we can
        do with a minimal list walker implementation.
        