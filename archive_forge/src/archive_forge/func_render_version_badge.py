from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def render_version_badge(pypi_exists, local_version, name):
    """Renders a version badge for the package. PyPi badge if it exists, otherwise a static badge."""
    if pypi_exists:
        return f'<a href="https://pypi.org/project/{name}/" target="_blank"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/{name}"></a>'
    else:
        return f'<img alt="Static Badge" src="https://img.shields.io/badge/version%20-%20{local_version}%20-%20orange">'