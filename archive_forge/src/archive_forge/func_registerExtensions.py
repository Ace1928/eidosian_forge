from __future__ import annotations
import codecs
import sys
import logging
import importlib
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, ClassVar, Mapping, Sequence
from . import util
from .preprocessors import build_preprocessors
from .blockprocessors import build_block_parser
from .treeprocessors import build_treeprocessors
from .inlinepatterns import build_inlinepatterns
from .postprocessors import build_postprocessors
from .extensions import Extension
from .serializers import to_html_string, to_xhtml_string
from .util import BLOCK_LEVEL_ELEMENTS
def registerExtensions(self, extensions: Sequence[Extension | str], configs: Mapping[str, dict[str, Any]]) -> Markdown:
    """
        Load a list of extensions into an instance of the `Markdown` class.

        Arguments:
            extensions (list[Extension | str]): A list of extensions.

                If an item is an instance of a subclass of [`markdown.extensions.Extension`][],
                the instance will be used as-is. If an item is of type `str`, it is passed
                to [`build_extension`][markdown.Markdown.build_extension] with its corresponding `configs` and the
                returned instance  of [`markdown.extensions.Extension`][] is used.
            configs (dict[str, dict[str, Any]]): Configuration settings for extensions.

        """
    for ext in extensions:
        if isinstance(ext, str):
            ext = self.build_extension(ext, configs.get(ext, {}))
        if isinstance(ext, Extension):
            ext.extendMarkdown(self)
            logger.debug('Successfully loaded extension "%s.%s".' % (ext.__class__.__module__, ext.__class__.__name__))
        elif ext is not None:
            raise TypeError('Extension "{}.{}" must be of type: "{}.{}"'.format(ext.__class__.__module__, ext.__class__.__name__, Extension.__module__, Extension.__name__))
    return self