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
def handle_java_script_dialog(accept: bool, prompt_text: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Accepts or dismisses a JavaScript initiated dialog (alert, confirm, prompt, or onbeforeunload).

    :param accept: Whether to accept or dismiss the dialog.
    :param prompt_text: *(Optional)* The text to enter into the dialog prompt before accepting. Used only if this is a prompt dialog.
    """
    params: T_JSON_DICT = dict()
    params['accept'] = accept
    if prompt_text is not None:
        params['promptText'] = prompt_text
    cmd_dict: T_JSON_DICT = {'method': 'Page.handleJavaScriptDialog', 'params': params}
    json = (yield cmd_dict)