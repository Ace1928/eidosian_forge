from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
def automove_cursor() -> None:
    ch = 0
    last_hidden = False
    first_visible = False
    for pwi, (w, _o) in enumerate(ow.contents):
        wcanv = w.render((maxcol,))
        wh = wcanv.rows()
        if wh:
            ch += wh
        if not last_hidden and ch >= self._trim_top:
            last_hidden = True
        elif last_hidden:
            if not first_visible:
                first_visible = True
            if not w.selectable():
                continue
            ow.focus_item = pwi
            st = None
            nf = ow.get_focus()
            if hasattr(nf, 'key_timeout'):
                st = nf
            elif hasattr(nf, 'original_widget'):
                no = nf.original_widget
                if hasattr(no, 'original_widget'):
                    st = no.original_widget
                elif hasattr(no, 'key_timeout'):
                    st = no
            if st and hasattr(st, 'key_timeout') and callable(getattr(st, 'keypress', None)):
                st.keypress(None, None)
            break