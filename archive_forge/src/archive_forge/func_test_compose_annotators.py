from holoviews import Overlay
from holoviews.annotators import PathAnnotator, PointAnnotator, annotate
from holoviews.element import Path, Points, Table
from holoviews.element.tiles import EsriStreet, Tiles
from holoviews.tests.plotting.bokeh.test_plot import TestBokehPlot
def test_compose_annotators(self):
    layout1 = annotate(Points([]), annotations=['Label'])
    layout2 = annotate(Path([]), annotations=['Name'])
    combined = annotate.compose(layout1, layout2)
    overlay = combined.DynamicMap.I[()]
    tables = combined.Annotator.I[()]
    self.assertIsInstance(overlay, Overlay)
    self.assertEqual(len(overlay), 2)
    self.assertEqual(overlay.get(0), Points([], vdims='Label'))
    self.assertEqual(overlay.get(1), Path([], vdims='Name'))
    self.assertIsInstance(tables, Overlay)
    self.assertEqual(len(tables), 3)