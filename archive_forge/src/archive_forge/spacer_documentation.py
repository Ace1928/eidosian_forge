from __future__ import annotations
from typing import ClassVar, List
import param
from bokeh.models import Div as BkDiv, Spacer as BkSpacer
from ..io.resources import CDN_DIST
from ..reactive import Reactive

    A `Divider` draws a horizontal rule (a `<hr>` tag in HTML) to separate
    multiple components in a layout. It automatically spans the full width of
    the container.

    Reference: https://panel.holoviz.org/reference/layouts/Divider.html

    :Example:

    >>> pn.Column(
    ...     '# Lorem Ipsum',
    ...     pn.layout.Divider(),
    ...     'A very long text... '
    >>> )
    