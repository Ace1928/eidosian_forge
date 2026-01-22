import pytest
from playwright.sync_api import expect
from panel.config import config
from panel.io.state import state
from panel.pane import Markdown
from panel.template import BootstrapTemplate
from panel.tests.util import serve_component
from panel.widgets import Button
def test_ready_notification(page):

    def app():
        config.ready_notification = 'Ready!'
        return Markdown('Ready app')
    serve_component(page, app)
    expect(page.locator('.notyf__message')).to_have_text('Ready!')