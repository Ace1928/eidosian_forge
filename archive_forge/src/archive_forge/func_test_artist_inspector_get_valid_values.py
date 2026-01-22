import io
from itertools import chain
import numpy as np
import pytest
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.artist as martist
import matplotlib.backend_bases as mbackend_bases
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@pytest.mark.parametrize('accept_clause, expected', [('', 'unknown'), ("ACCEPTS: [ '-' | '--' | '-.' ]", "[ '-' | '--' | '-.' ]"), ('ACCEPTS: Some description.', 'Some description.'), ('.. ACCEPTS: Some description.', 'Some description.'), ('arg : int', 'int'), ('*arg : int', 'int'), ('arg : int\nACCEPTS: Something else.', 'Something else. ')])
def test_artist_inspector_get_valid_values(accept_clause, expected):

    class TestArtist(martist.Artist):

        def set_f(self, arg):
            pass
    TestArtist.set_f.__doc__ = '\n    Some text.\n\n    %s\n    ' % accept_clause
    valid_values = martist.ArtistInspector(TestArtist).get_valid_values('f')
    assert valid_values == expected