import pandas as pd
import pytest
from panel.pane import Perspective
from panel.tests.util import serve_component, wait_until
def test_perspective_click_event(page, perspective_data):
    perspective = Perspective(perspective_data)
    events = []
    perspective.on_click(lambda e: events.append(e))
    serve_component(page, perspective)
    page.locator('.pnx-perspective-viewer').locator('tr').nth(4).locator('td').nth(3).click(force=True)
    wait_until(lambda: len(events) == 1, page)
    event = events[0]
    assert event.config == {'filter': []}
    assert event.column_names == ['C']
    assert event.row == {'index': 2, 'A': 2, 'B': 0, 'C': 'foo3', 'D': 1231113600000}