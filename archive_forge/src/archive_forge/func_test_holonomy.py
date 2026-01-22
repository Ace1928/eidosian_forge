from ..sage_helper import _within_sage, doctest_modules
from ..pari import pari
import snappy
import snappy.snap as snap
import getopt
import sys
def test_holonomy(dec_prec=200):

    def test_manifold(manifold):
        G = snap.polished_holonomy(manifold, dec_prec=dec_prec)
    for census in [snappy.OrientableCuspedCensus, snappy.OrientableClosedCensus]:
        print('Testing holonomy of 100 manifolds in ', census)
        for manifold in census()[-100:]:
            test_manifold(manifold)