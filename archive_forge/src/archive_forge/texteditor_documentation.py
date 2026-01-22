from __future__ import annotations
from typing import (
import param
from pyviz_comms import JupyterComm
from ..util import lazy_load
from .base import Widget

    The `TextEditor` widget provides a WYSIWYG
    (what-you-see-is-what-you-get) rich text editor which outputs HTML.

    The editor is built on top of the [Quill.js](https://quilljs.com/) library.

    Reference: https://panel.holoviz.org/reference/widgets/TextEditor.html

    :Example:

    >>> TextEditor(placeholder='Enter some text')
    