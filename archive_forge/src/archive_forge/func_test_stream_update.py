from holoviews import Overlay
from holoviews.annotators import PathAnnotator, PointAnnotator, annotate
from holoviews.element import Path, Points, Table
from holoviews.element.tiles import EsriStreet, Tiles
from holoviews.tests.plotting.bokeh.test_plot import TestBokehPlot
def test_stream_update(self):
    annotator = PointAnnotator(Points([(1, 2)]), annotations=['Label'])
    annotator._stream.event(data={'x': [1], 'y': [2], 'Label': ['A']})
    self.assertEqual(annotator.object, Points([(1, 2, 'A')], vdims=['Label']))