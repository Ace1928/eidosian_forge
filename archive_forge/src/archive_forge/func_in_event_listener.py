from __future__ import annotations
import ast
import inspect
from abc import ABCMeta
from functools import wraps
from pathlib import Path
from jinja2 import Template
from gradio.events import EventListener
from gradio.exceptions import ComponentDefinitionError
from gradio.utils import no_raise_exception
def in_event_listener():
    from gradio.context import LocalContext
    return LocalContext.in_event_listener.get()