import pytest
from bokeh.models import Tooltip
from playwright.sync_api import Expect, expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
@pytest.mark.parametrize('button_fn,button_locator', [(lambda **kw: Button(**kw), '.bk-btn'), (lambda **kw: CheckButtonGroup(options=['A', 'B'], **kw), '.bk-btn-group'), (lambda **kw: RadioButtonGroup(options=['A', 'B'], **kw), '.bk-btn-group')], ids=['Button', 'CheckButtonGroup', 'RadioButtonGroup'])
def test_button_tooltip_with_delay(page, button_fn, button_locator):
    pn_button = button_fn(name='test', description='Test', description_delay=300)
    exp = Expect()
    exp.set_options(timeout=200)
    serve_component(page, pn_button)
    button = page.locator(button_locator)
    expect(button).to_have_count(1)
    tooltip = page.locator('.bk-tooltip-content')
    expect(tooltip).to_have_count(0)
    page.hover(button_locator)
    tooltip = page.locator('.bk-tooltip-content')
    exp(tooltip).to_have_count(0)
    page.wait_for_timeout(200)
    exp(tooltip).to_have_count(1)
    page.hover('body')
    tooltip = page.locator('.bk-tooltip-content')
    exp(tooltip).to_have_count(0)
    page.hover(button_locator)
    page.wait_for_timeout(50)
    page.hover('body')
    page.wait_for_timeout(300)
    exp(tooltip).to_have_count(0)