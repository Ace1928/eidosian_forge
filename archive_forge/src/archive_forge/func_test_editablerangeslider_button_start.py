import pytest
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
@pytest.mark.parametrize('widget', [EditableRangeSlider], ids=['EditableRangeSlider'])
def test_editablerangeslider_button_start(page, widget):
    widget = widget(step=1)
    default_value = widget.value
    step = widget.step
    start = widget.start
    serve_component(page, widget)
    up = page.locator('button').nth(0)
    down = page.locator('button').nth(1)
    down.click()
    wait_until(lambda: widget.value == (default_value[0] - step, 1), page)
    wait_until(lambda: widget._slider.start == start - step, page)
    down.click()
    wait_until(lambda: widget.value == (default_value[0] - 2 * step, 1), page)
    wait_until(lambda: widget._slider.start == start - 2 * step, page)
    up.click()
    wait_until(lambda: widget.value == (default_value[0] - step, 1), page)
    wait_until(lambda: widget._slider.start == start - 2 * step, page)