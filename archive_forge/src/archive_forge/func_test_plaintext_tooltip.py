import pytest
from bokeh.models import Tooltip
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import TooltipIcon
@pytest.mark.parametrize('value', ['Test', Tooltip(content='Test', position='right')], ids=['str', 'Tooltip'])
def test_plaintext_tooltip(page, value):
    tooltip_icon = TooltipIcon(value='Test')
    serve_component(page, tooltip_icon)
    icon = page.locator('.bk-icon')
    expect(icon).to_have_count(1)
    tooltip = page.locator('.bk-tooltip-content')
    expect(tooltip).to_have_count(0)
    page.hover('.bk-icon')
    tooltip = page.locator('.bk-tooltip-content')
    expect(tooltip).to_have_count(1)
    expect(tooltip).to_have_text('Test')
    page.hover('body')
    tooltip = page.locator('.bk-tooltip-content')
    expect(tooltip).to_have_count(0)
    page.click('.bk-icon')
    tooltip = page.locator('.bk-tooltip-content')
    expect(tooltip).to_have_count(1)
    page.hover('body')
    tooltip = page.locator('.bk-tooltip-content')
    expect(tooltip).to_have_count(1)
    page.click('body')
    tooltip = page.locator('.bk-tooltip-content')
    expect(tooltip).to_have_count(0)