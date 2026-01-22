import unittest
from panel.config import config
from panel.io.notifications import NotificationArea
from panel.io.state import set_curdoc, state
from panel.template import VanillaTemplate
from panel.widgets import Button
def test_template_multi_refs():
    tmpl = VanillaTemplate()
    button = Button(name='Click me', button_type='primary')
    tmpl.sidebar.append(button)
    tmpl.main.append(button)
    assert len(tmpl._render_items) == 6
    assert f'nav-{id(button)}' in tmpl._render_items
    assert f'main-{id(button)}' in tmpl._render_items