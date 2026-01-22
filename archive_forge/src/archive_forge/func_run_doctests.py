from ..sage_helper import _within_sage, doctest_modules
from ..pari import pari
import snappy
import snappy.snap as snap
import getopt
import sys
def run_doctests(verbose=False, print_info=True):
    from snappy.snap.t3mlite import perm4
    from snappy.snap.t3mlite import linalg
    from snappy.snap.t3mlite import spun
    from snappy.snap.t3mlite import mcomplex
    from snappy.snap import slice_obs_HKL
    from snappy.snap import character_varieties
    from snappy.snap import nsagetools
    from snappy.snap import polished_reps
    from snappy.snap import interval_reps
    from snappy.snap import fundamental_polyhedron
    from snappy.snap.peripheral import dual_cellulation
    from snappy.snap.peripheral import link
    from snappy.snap.peripheral import peripheral
    modules = [perm4, mcomplex, linalg, spun, character_varieties, nsagetools, slice_obs_HKL, polished_reps, snap, interval_reps, fundamental_polyhedron, dual_cellulation, link, peripheral]
    globs = {'Manifold': snappy.Manifold, 'ManifoldHP': snappy.ManifoldHP, 'Triangulation': snappy.Triangulation, 'Mcomplex': snappy.snap.t3mlite.Mcomplex, 'LinkSurface': snappy.snap.peripheral.link.LinkSurface}
    return doctest_modules(modules, extraglobs=globs, verbose=verbose, print_info=print_info)