import pytest
from numpy.testing import assert_allclose
from panel.layout import Row
from panel.pane import PDF, PNG, SVG
from panel.tests.util import serve_component, wait_for_server
def test_pdf_no_embed_start_page(page):
    pdf_pane = PDF(PDF_FILE, start_page=22, embed=False)
    serve_component(page, pdf_pane)
    src = page.locator('embed').get_attribute('src')
    assert src == PDF_FILE + '#page=22'