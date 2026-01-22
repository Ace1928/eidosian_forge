import pytest
from numpy.testing import assert_allclose, assert_array_equal
from matplotlib.sankey import Sankey
from matplotlib.testing.decorators import check_figures_equal
def test_sankey2():
    s = Sankey(flows=[0.25, -0.25, 0.5, -0.5], labels=['Foo'], orientations=[-1], unit='Bar')
    sf = s.finish()
    assert_array_equal(sf[0].flows, [0.25, -0.25, 0.5, -0.5])
    assert sf[0].angles == [1, 3, 1, 3]
    assert all([text.get_text()[0:3] == 'Foo' for text in sf[0].texts])
    assert all([text.get_text()[-3:] == 'Bar' for text in sf[0].texts])
    assert sf[0].text.get_text() == ''
    assert_allclose(sf[0].tips, [(-1.375, -0.52011255), (1.375, -0.75506044), (-0.75, -0.41522509), (0.75, -0.8599479)])
    s = Sankey(flows=[0.25, -0.25, 0, 0.5, -0.5], labels=['Foo'], orientations=[-1], unit='Bar')
    sf = s.finish()
    assert_array_equal(sf[0].flows, [0.25, -0.25, 0, 0.5, -0.5])
    assert sf[0].angles == [1, 3, None, 1, 3]
    assert_allclose(sf[0].tips, [(-1.375, -0.52011255), (1.375, -0.75506044), (0, 0), (-0.75, -0.41522509), (0.75, -0.8599479)])