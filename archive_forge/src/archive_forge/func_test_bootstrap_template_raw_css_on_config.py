import pytest
from playwright.sync_api import expect
from panel.pane import Markdown
from panel.template import BootstrapTemplate
from panel.tests.util import serve_component
def test_bootstrap_template_raw_css_on_config(page):
    tmpl = BootstrapTemplate()
    tmpl.config.raw_css = ['.markdown { color: rgb(255, 0, 0); }']
    md = Markdown('Initial')
    tmpl.main.append(md)
    msgs, _ = serve_component(page, tmpl)
    expect(page.locator('.markdown')).to_have_css('color', 'rgb(255, 0, 0)')
    assert [msg for msg in msgs if msg.type == 'error'] == []