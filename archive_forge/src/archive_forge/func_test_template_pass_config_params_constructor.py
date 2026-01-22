import unittest
from panel.config import config
from panel.io.notifications import NotificationArea
from panel.io.state import set_curdoc, state
from panel.template import VanillaTemplate
from panel.widgets import Button
def test_template_pass_config_params_constructor(document):
    custom_config = {'raw_css': ['html { background-color: purple; }'], 'css_files': ['stylesheet.css'], 'js_files': {'foo': 'foo.js'}, 'js_modules': {'bar': 'bar.js'}}
    tmpl = VanillaTemplate(**custom_config)
    config = tmpl.config.param.values()
    del config['name']
    assert config == custom_config