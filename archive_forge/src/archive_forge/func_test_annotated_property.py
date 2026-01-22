from holoviews import Overlay
from holoviews.annotators import PathAnnotator, PointAnnotator, annotate
from holoviews.element import Path, Points, Table
from holoviews.element.tiles import EsriStreet, Tiles
from holoviews.tests.plotting.bokeh.test_plot import TestBokehPlot
def test_annotated_property(self):
    annotator = annotate.instance()
    annotator(Points([]), annotations=['Label'])
    self.assertIn('Label', annotator.annotated)