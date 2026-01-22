import time
import pytest
from playwright.sync_api import expect
from panel.pane import panel
from panel.tests.util import serve_component
def defer_load():
    time.sleep(0.5)
    return 'I render after load!'