from os import path
from typing import Any, Dict, cast
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.domains.math import MathDomain
from sphinx.environment import BuildEnvironment
from sphinx.errors import ExtensionError
from sphinx.locale import get_translation
from sphinx.util.math import get_node_equation_number
from sphinx.writers.html import HTMLTranslator
from sphinxcontrib.jsmath.version import __version__
def install_jsmath(app: Sphinx, env: BuildEnvironment) -> None:
    if app.builder.format != 'html' or app.builder.math_renderer_name != 'jsmath':
        return
    if not app.config.jsmath_path:
        raise ExtensionError('jsmath_path config value must be set for the jsmath extension to work')
    builder = cast(StandaloneHTMLBuilder, app.builder)
    domain = cast(MathDomain, env.get_domain('math'))
    if domain.has_equations():
        builder.add_js_file(app.config.jsmath_path)