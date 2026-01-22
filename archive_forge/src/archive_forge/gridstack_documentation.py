from __future__ import annotations
from typing import ClassVar, List, Mapping
import param
from ..config import config
from ..io.resources import CDN_DIST, bundled_files
from ..reactive import ReactiveHTML
from ..util import classproperty
from .grid import GridSpec

    The `GridStack` layout allows arranging multiple Panel objects in a grid
    using a simple API to assign objects to individual grid cells or to a grid
    span.

    Other layout containers function like lists, but a `GridSpec` has an API
    similar to a 2D array, making it possible to use 2D assignment to populate,
    index, and slice the grid.

    Reference: https://panel.holoviz.org/reference/layouts/GridStack.html

    :Example:

    >>> pn.extension('gridstack')
    >>> gstack = GridStack(sizing_mode='stretch_both')
    >>> gstack[ : , 0: 3] = pn.Spacer(styles=dict(background='red'))
    >>> gstack[0:2, 3: 9] = pn.Spacer(styles=dict(background='green'))
    >>> gstack[2:4, 6:12] = pn.Spacer(styles=dict(background='orange'))
    >>> gstack[4:6, 3:12] = pn.Spacer(styles=dict(background='blue'))
    >>> gstack[0:2, 9:12] = pn.Spacer(styles=dict(background='purple'))
    