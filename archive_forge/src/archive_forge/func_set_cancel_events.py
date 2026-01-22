from __future__ import annotations
import dataclasses
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence
from gradio_client.documentation import document
from jinja2 import Template
from gradio.context import Context
from gradio.utils import get_cancel_function
def set_cancel_events(triggers: Sequence[EventListenerMethod], cancels: None | dict[str, Any] | list[dict[str, Any]]):
    if cancels:
        if not isinstance(cancels, list):
            cancels = [cancels]
        cancel_fn, fn_indices_to_cancel = get_cancel_function(cancels)
        if Context.root_block is None:
            raise AttributeError('Cannot cancel outside of a gradio.Blocks context.')
        Context.root_block.set_event_trigger(triggers, cancel_fn, inputs=None, outputs=None, queue=False, preprocess=False, show_api=False, cancels=fn_indices_to_cancel)