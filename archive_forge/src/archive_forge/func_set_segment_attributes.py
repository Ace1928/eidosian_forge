from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
def set_segment_attributes(self, attlist, hatt=None, satt=None):
    """
        :param attlist: list containing display attribute or
                        (display attribute, character) tuple for background,
                        first segment, and optionally following segments.
                        ie. len(attlist) == num segments+1
                        character defaults to ' ' if not specified.
        :param hatt: list containing attributes for horizontal lines. First
                     element is for lines on background, second is for lines
                     on first segment, third is for lines on second segment
                     etc.
        :param satt: dictionary containing attributes for smoothed
                     transitions of bars in UTF-8 display mode. The values
                     are in the form:

                       (fg,bg) : attr

                     fg and bg are integers where 0 is the graph background,
                     1 is the first segment, 2 is the second, ...
                     fg > bg in all values.  attr is an attribute with a
                     foreground corresponding to fg and a background
                     corresponding to bg.

        If satt is not None and the bar graph is being displayed in
        a terminal using the UTF-8 encoding then the character cell
        that is shared between the segments specified will be smoothed
        with using the UTF-8 vertical eighth characters.

        eg: set_segment_attributes( ['no', ('unsure',"?"), 'yes'] )
        will use the attribute 'no' for the background (the area from
        the top of the graph to the top of the bar), question marks
        with the attribute 'unsure' will be used for the topmost
        segment of the bar, and the attribute 'yes' will be used for
        the bottom segment of the bar.
        """
    self.attr = []
    self.char = []
    if len(attlist) < 2:
        raise BarGraphError(f'attlist must include at least background and seg1: {attlist!r}')
    if len(attlist) < 2:
        raise BarGraphError('must at least specify bg and fg!')
    for a in attlist:
        if not isinstance(a, tuple):
            self.attr.append(a)
            self.char.append(' ')
        else:
            attr, ch = a
            self.attr.append(attr)
            self.char.append(ch)
    self.hatt = []
    if hatt is None:
        hatt = [self.attr[0]]
    elif not isinstance(hatt, list):
        hatt = [hatt]
    self.hatt = hatt
    if satt is None:
        satt = {}
    for i in satt.items():
        try:
            (fg, bg), attr = i
        except ValueError as exc:
            raise BarGraphError(f'satt not in (fg,bg:attr) form: {i!r}').with_traceback(exc.__traceback__) from exc
        if not isinstance(fg, int) or fg >= len(attlist):
            raise BarGraphError(f'fg not valid integer: {fg!r}')
        if not isinstance(bg, int) or bg >= len(attlist):
            raise BarGraphError(f'bg not valid integer: {fg!r}')
        if fg <= bg:
            raise BarGraphError(f'fg ({fg}) not > bg ({bg})')
    self.satt = satt