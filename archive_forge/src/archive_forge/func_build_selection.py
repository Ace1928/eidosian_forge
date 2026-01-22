import numpy as np
from ...core.options import Store
from ...core.overlay import NdOverlay, Overlay
from ...selection import OverlaySelectionDisplay, SelectionDisplay
def build_selection(self, selection_streams, hvobj, operations, region_stream=None, cache=None):
    if cache is None:
        cache = {}
    sel_streams = [selection_streams.exprs_stream]
    hvobj = hvobj.apply(self._build_selection, streams=sel_streams, per_element=True)
    for op in operations:
        hvobj = op(hvobj)
    return hvobj