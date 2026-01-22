import pytest
from playwright.sync_api import expect
from panel.config import config
from panel.io.state import state
from panel.pane import Markdown
from panel.template import BootstrapTemplate
from panel.tests.util import serve_component
from panel.widgets import Button
def test_disconnect_notification(page):

    def app():
        config.disconnect_notification = 'Disconnected!'
        button = Button(name='Stop server')
        button.js_on_click(code="\n        Bokeh.documents[0].event_manager.send_event({'event_name': 'connection_lost', 'publish': false})\n        ")
        return button
    serve_component(page, app)
    page.click('.bk-btn')
    expect(page.locator('.notyf__message')).to_have_text('Disconnected!')