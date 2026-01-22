from os import path
from re import DOTALL, match
from textwrap import indent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, TypeVar
from docutils import nodes
from docutils.io import StringInput
from docutils.nodes import Element
from docutils.utils import relative_path
from sphinx import addnodes
from sphinx.config import Config
from sphinx.domains.std import make_glossary_term, split_term_classifiers
from sphinx.locale import __
from sphinx.locale import init as init_locale
from sphinx.transforms import SphinxTransform
from sphinx.util import get_filetype, logging, split_index_msg
from sphinx.util.i18n import docname_to_domain
from sphinx.util.nodes import (IMAGE_TYPE_NODES, LITERAL_TYPE_NODES, NodeMatcher,
def publish_msgstr(app: 'Sphinx', source: str, source_path: str, source_line: int, config: Config, settings: Any) -> Element:
    """Publish msgstr (single line) into docutils document

    :param sphinx.application.Sphinx app: sphinx application
    :param str source: source text
    :param str source_path: source path for warning indication
    :param source_line: source line for warning indication
    :param sphinx.config.Config config: sphinx config
    :param docutils.frontend.Values settings: docutils settings
    :return: document
    :rtype: docutils.nodes.document
    """
    try:
        rst_prolog = config.rst_prolog
        config.rst_prolog = None
        from sphinx.io import SphinxI18nReader
        reader = SphinxI18nReader()
        reader.setup(app)
        filetype = get_filetype(config.source_suffix, source_path)
        parser = app.registry.create_source_parser(app, filetype)
        doc = reader.read(source=StringInput(source=source, source_path='%s:%s:<translated>' % (source_path, source_line)), parser=parser, settings=settings)
        try:
            doc = doc[0]
        except IndexError:
            pass
        return doc
    finally:
        config.rst_prolog = rst_prolog