from __future__ import annotations
import typing
from urwid import text_layout
from urwid.canvas import apply_text_layout
from urwid.split_repr import remove_defaults
from urwid.str_util import calc_width
from urwid.util import decompose_tagmarkup, get_encoding
from .constants import Align, Sizing, WrapMode
from .widget import Widget, WidgetError
def set_wrap_mode(self, mode: Literal['space', 'any', 'clip', 'ellipsis'] | WrapMode) -> None:
    """
        Set text wrapping mode. Supported modes depend on text layout
        object in use but defaults to a :class:`StandardTextLayout` instance

        :param mode: typically ``'space'``, ``'any'``, ``'clip'`` or ``'ellipsis'``
        :type mode: text wrapping mode

        >>> t = Text(u"some words")
        >>> t.render((6,)).text # ... = b in Python 3
        [...'some  ', ...'words ']
        >>> t.set_wrap_mode('clip')
        >>> t.wrap
        'clip'
        >>> t.render((6,)).text
        [...'some w']
        >>> t.wrap = 'any'  # Urwid 0.9.9 or later
        >>> t.render((6,)).text
        [...'some w', ...'ords  ']
        >>> t.wrap = 'somehow'
        Traceback (most recent call last):
        TextError: Wrap mode 'somehow' not supported.
        """
    if not self.layout.supports_wrap_mode(mode):
        raise TextError(f'Wrap mode {mode!r} not supported.')
    self._wrap_mode = mode
    self._invalidate()