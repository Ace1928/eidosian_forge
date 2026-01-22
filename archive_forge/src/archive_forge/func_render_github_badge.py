from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def render_github_badge(repo):
    """Renders a github badge for the package if a repo is specified."""
    if repo is None:
        return ''
    else:
        return f'<a href="{repo}/issues" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Issues-white?logo=github&logoColor=black"></a>'