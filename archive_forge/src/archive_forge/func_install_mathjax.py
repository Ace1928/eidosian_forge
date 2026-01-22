import json
from typing import Any, Dict, cast
from docutils import nodes
import sphinx
from sphinx.application import Sphinx
from sphinx.domains.math import MathDomain
from sphinx.errors import ExtensionError
from sphinx.locale import _
from sphinx.util.math import get_node_equation_number
from sphinx.writers.html import HTMLTranslator
def install_mathjax(app: Sphinx, pagename: str, templatename: str, context: Dict[str, Any], event_arg: Any) -> None:
    if app.builder.format != 'html' or app.builder.math_renderer_name != 'mathjax':
        return
    if not app.config.mathjax_path:
        raise ExtensionError('mathjax_path config value must be set for the mathjax extension to work')
    domain = cast(MathDomain, app.env.get_domain('math'))
    if app.registry.html_assets_policy == 'always' or domain.has_equations(pagename):
        if app.config.mathjax2_config:
            if app.config.mathjax_path == MATHJAX_URL:
                logger.warning('mathjax_config/mathjax2_config does not work for the current MathJax version, use mathjax3_config instead')
            body = 'MathJax.Hub.Config(%s)' % json.dumps(app.config.mathjax2_config)
            app.add_js_file(None, type='text/x-mathjax-config', body=body)
        if app.config.mathjax3_config:
            body = 'window.MathJax = %s' % json.dumps(app.config.mathjax3_config)
            app.add_js_file(None, body=body)
        options = {}
        if app.config.mathjax_options:
            options.update(app.config.mathjax_options)
        if 'async' not in options and 'defer' not in options:
            if app.config.mathjax3_config:
                options['defer'] = 'defer'
            else:
                options['async'] = 'async'
        app.add_js_file(app.config.mathjax_path, **options)