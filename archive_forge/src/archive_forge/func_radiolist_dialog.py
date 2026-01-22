from __future__ import annotations
import functools
from asyncio import get_running_loop
from typing import Any, Callable, Sequence, TypeVar
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.eventloop import run_in_executor_with_context
from prompt_toolkit.filters import FilterOrBool
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.key_binding.key_bindings import KeyBindings, merge_key_bindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import AnyContainer, HSplit
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.styles import BaseStyle
from prompt_toolkit.validation import Validator
from prompt_toolkit.widgets import (
def radiolist_dialog(title: AnyFormattedText='', text: AnyFormattedText='', ok_text: str='Ok', cancel_text: str='Cancel', values: Sequence[tuple[_T, AnyFormattedText]] | None=None, default: _T | None=None, style: BaseStyle | None=None) -> Application[_T]:
    """
    Display a simple list of element the user can choose amongst.

    Only one element can be selected at a time using Arrow keys and Enter.
    The focus can be moved between the list and the Ok/Cancel button with tab.
    """
    if values is None:
        values = []

    def ok_handler() -> None:
        get_app().exit(result=radio_list.current_value)
    radio_list = RadioList(values=values, default=default)
    dialog = Dialog(title=title, body=HSplit([Label(text=text, dont_extend_height=True), radio_list], padding=1), buttons=[Button(text=ok_text, handler=ok_handler), Button(text=cancel_text, handler=_return_none)], with_background=True)
    return _create_app(dialog, style)