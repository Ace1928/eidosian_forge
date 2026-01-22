from __future__ import annotations
import collections
import copy
import datetime
import os
import sys
import typing as t
import nbformat
from nbformat import NotebookNode, validator
from traitlets import Bool, HasTraits, List, TraitError, Unicode
from traitlets.config import Config
from traitlets.config.configurable import LoggingConfigurable
from traitlets.utils.importstring import import_item
def register_preprocessor(self, preprocessor, enabled=False):
    """
        Register a preprocessor.
        Preprocessors are classes that act upon the notebook before it is
        passed into the Jinja templating engine. Preprocessors are also
        capable of passing additional information to the Jinja
        templating engine.

        Parameters
        ----------
        preprocessor : `nbconvert.preprocessors.Preprocessor`
            A dotted module name, a type, or an instance
        enabled : bool
            Mark the preprocessor as enabled

        """
    if preprocessor is None:
        msg = 'preprocessor must not be None'
        raise TypeError(msg)
    isclass = isinstance(preprocessor, type)
    constructed = not isclass
    if constructed and isinstance(preprocessor, str):
        preprocessor_cls = import_item(preprocessor)
        return self.register_preprocessor(preprocessor_cls, enabled)
    if constructed and callable(preprocessor):
        if enabled:
            preprocessor.enabled = True
        self._preprocessors.append(preprocessor)
        return preprocessor
    if isclass and issubclass(preprocessor, HasTraits):
        self.register_preprocessor(preprocessor(parent=self), enabled)
        return None
    if isclass:
        self.register_preprocessor(preprocessor(), enabled)
        return None
    raise TypeError('preprocessor must be callable or an importable constructor, got %r' % preprocessor)