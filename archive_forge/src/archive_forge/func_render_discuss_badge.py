from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def render_discuss_badge(space):
    """Renders a discuss badge for the package if a space is specified."""
    if space is None:
        return ''
    else:
        return f'<a href="{space}/discussions" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97%20Discuss-%23097EFF?style=flat&logoColor=black"></a>'