from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def format_none(value):
    """Formats None and NonType values."""
    if value is None or value is type(None) or value == 'None' or (value == 'NoneType'):
        return 'None'
    return value