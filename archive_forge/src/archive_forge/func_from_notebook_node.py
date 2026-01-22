from __future__ import annotations
import html
import json
import os
import typing as t
import uuid
import warnings
from pathlib import Path
from jinja2 import (
from jupyter_core.paths import jupyter_path
from nbformat import NotebookNode
from traitlets import Bool, Dict, HasTraits, List, Unicode, default, observe, validate
from traitlets.config import Config
from traitlets.utils.importstring import import_item
from nbconvert import filters
from .exporter import Exporter
def from_notebook_node(self, nb: NotebookNode, resources: dict[str, t.Any] | None=None, **kw: t.Any) -> tuple[str, dict[str, t.Any]]:
    """
        Convert a notebook from a notebook node instance.

        Parameters
        ----------
        nb : :class:`~nbformat.NotebookNode`
            Notebook node
        resources : dict
            Additional resources that can be accessed read/write by
            preprocessors and filters.
        """
    nb_copy, resources = super().from_notebook_node(nb, resources, **kw)
    resources.setdefault('raw_mimetypes', self.raw_mimetypes)
    resources['global_content_filter'] = {'include_code': not self.exclude_code_cell, 'include_markdown': not self.exclude_markdown, 'include_raw': not self.exclude_raw, 'include_unknown': not self.exclude_unknown, 'include_input': not self.exclude_input, 'include_output': not self.exclude_output, 'include_output_stdin': not self.exclude_output_stdin, 'include_input_prompt': not self.exclude_input_prompt, 'include_output_prompt': not self.exclude_output_prompt, 'no_prompt': self.exclude_input_prompt and self.exclude_output_prompt}
    output = self.template.render(nb=nb_copy, resources=resources)
    output = output.lstrip('\r\n')
    return (output, resources)