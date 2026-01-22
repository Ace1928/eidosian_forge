from __future__ import annotations
import asyncio
import glob
import logging
import os
import sys
import typing as t
from textwrap import dedent, fill
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, DottedObjectName, Instance, List, Type, Unicode, default, observe
from traitlets.config import Configurable, catch_config_error
from traitlets.utils.importstring import import_item
from nbconvert import __version__, exporters, postprocessors, preprocessors, writers
from nbconvert.utils.text import indent
from .exporters.base import get_export_names, get_exporter
from .utils.base import NbConvertBase
from .utils.exceptions import ConversionException
from .utils.io import unicode_stdin_stream
def export_single_notebook(self, notebook_filename, resources, input_buffer=None):
    """Step 2: Export the notebook

        Exports the notebook to a particular format according to the specified
        exporter. This function returns the output and (possibly modified)
        resources from the exporter.

        Parameters
        ----------
        notebook_filename : str
            name of notebook file.
        resources : dict
        input_buffer :
            readable file-like object returning unicode.
            if not None, notebook_filename is ignored

        Returns
        -------
        output
        dict
            resources (possibly modified)
        """
    try:
        if input_buffer is not None:
            output, resources = self.exporter.from_file(input_buffer, resources=resources)
        else:
            output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
    except ConversionException:
        self.log.error("Error while converting '%s'", notebook_filename, exc_info=True)
        self.exit(1)
    return (output, resources)