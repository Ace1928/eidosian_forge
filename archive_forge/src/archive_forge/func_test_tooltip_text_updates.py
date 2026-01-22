import pytest
from bokeh.models import Tooltip
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import TooltipIcon
def test_tooltip_text_updates(page):
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
    tooltip_icon.value = 'Updated'

    def hover():
        page.hover('.bk-icon')
        visible = page.locator('.bk-tooltip-content').count() == 1
        page.hover('body')
        return visible
    wait_until(hover, page)
    page.hover('.bk-icon')
    tooltip = page.locator('.bk-tooltip-content')
    expect(tooltip).to_have_count(1)
    expect(tooltip).to_have_text('Updated')