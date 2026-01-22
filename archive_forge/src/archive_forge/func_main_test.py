import re
import string
def main_test():
    import snappy
    censuses = [snappy.OrientableClosedCensus[:100], snappy.OrientableCuspedCensus(filter='tets<7'), snappy.NonorientableClosedCensus, snappy.NonorientableCuspedCensus, snappy.CensusKnots(), snappy.HTLinkExteriors(filter='cusps>3 and volume<14'), [snappy.Manifold(name) for name in asymmetric]]
    tests = 0
    for census in censuses:
        for M in census:
            isosig = decorated_isosig(M, snappy.Triangulation)
            N = snappy.Triangulation(isosig)
            assert same_peripheral_curves(M, N), M
            assert isosig == decorated_isosig(N, snappy.Triangulation), M
            assert M.homology() == N.homology()
            tests += 1
    print('Tested decorated isosig encode/decode on %d triangulations' % tests)