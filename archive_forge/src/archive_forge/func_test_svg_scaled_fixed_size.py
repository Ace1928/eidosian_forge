import pytest
from numpy.testing import assert_allclose
from panel.layout import Row
from panel.pane import PDF, PNG, SVG
from panel.tests.util import serve_component, wait_for_server
@pytest.mark.parametrize('embed', [False, True])
def test_svg_scaled_fixed_size(embed, page):
    svg = SVG(SVG_FILE, width=250, embed=embed)
    bbox = get_bbox(page, svg)
    assert bbox['width'] == 250
    assert int(bbox['height']) == 210