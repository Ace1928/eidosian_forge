import pytest
from panel import FloatPanel, Row, Spacer
from panel.tests.util import serve_component, wait_until
def test_float_panel_status_set_on_init(page):
    float_panel = FloatPanel(Spacer(styles=dict(background='red'), width=200, height=200), Spacer(styles=dict(background='green'), width=200, height=200), Spacer(styles=dict(background='blue'), width=200, height=200), status='minimized')
    serve_component(page, float_panel)
    float_container = page.locator('#jsPanel-replacement-container')
    wait_until(lambda: float_container.bounding_box()['y'] + float_container.bounding_box()['height'] == page.viewport_size['height'], page)