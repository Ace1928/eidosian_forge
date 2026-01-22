import numpy as np
import pandas as pd
from holoviews import render
from holoviews.core.data import Dataset, Dimension
from holoviews.element import Sankey
from .test_plot import TestBokehPlot, bokeh_renderer
def test_dimension_label(self):
    data = [['source1', 'dest1', 3], ['source1', 'dest2', 1]]
    df = pd.DataFrame(data, columns=['Source', 'Dest', 'Count'])
    kdims = [Dimension('Source'), Dimension('Dest', label='Dest Label')]
    plot = Sankey(df, kdims=kdims, vdims=['Count'])
    plot = plot.opts(edge_color='Dest Label')
    render(plot)