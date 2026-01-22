import pytest
from numpy.testing import assert_allclose, assert_array_equal
from matplotlib.sankey import Sankey
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('kwargs, msg', (({'trunklength': -1}, "'trunklength' is negative"), ({'flows': [0.2, 0.3], 'prior': 0}, 'The scaled sum of the connected'), ({'prior': -1}, 'The index of the prior diagram is negative'), ({'prior': 1}, 'The index of the prior diagram is 1'), ({'connect': (-1, 1), 'prior': 0}, 'At least one of the connection'), ({'connect': (2, 1), 'prior': 0}, 'The connection index to the source'), ({'connect': (1, 3), 'prior': 0}, 'The connection index to this dia'), ({'connect': (1, 1), 'prior': 0, 'flows': [-0.2, 0.2], 'orientations': [2]}, 'The value of orientations'), ({'connect': (1, 1), 'prior': 0, 'flows': [-0.2, 0.2], 'pathlengths': [2]}, "The lengths of 'flows'")))
def test_sankey_add_errors(kwargs, msg):
    sankey = Sankey()
    with pytest.raises(ValueError, match=msg):
        sankey.add(flows=[0.2, -0.2])
        sankey.add(**kwargs)