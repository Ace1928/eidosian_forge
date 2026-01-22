from __future__ import annotations
import typing as t
from collections import defaultdict
from textwrap import dedent
from traitlets import HasTraits, Undefined
from traitlets.config.application import Application
from traitlets.utils.text import indent
Write a rst file documenting config options for a traitlets application.

    Parameters
    ----------
    path : str
        The file to be written
    title : str
        The human-readable title of the document
    app : traitlets.config.Application
        An instance of the application class to be documented
    preamble : str
        Extra text to add just after the title (optional)
    