import pytest
from numpy.testing import assert_allclose
from panel.layout import Row
from panel.pane import PDF, PNG, SVG
from panel.tests.util import serve_component, wait_for_server
def test_pdf_embed(page):
    pdf_pane = PDF(PDF_FILE, embed=True)
    _, port = serve_component(page, pdf_pane)
    src = page.locator('embed').get_attribute('src')
    assert src.startswith(f'blob:http://localhost:{port}')
    assert src.endswith('#page=1')