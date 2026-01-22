import warnings
import plotly.graph_objs as go
from plotly.matplotlylib.mplexporter import Renderer
from plotly.matplotlylib import mpltools
def draw_legend_shapes(self, mode, shape, **props):
    """Create a shape that matches lines or markers in legends.

        Main issue is that path for circles do not render, so we have to use 'circle'
        instead of 'path'.
        """
    for single_mode in mode.split('+'):
        x = props['data'][0][0]
        y = props['data'][0][1]
        if single_mode == 'markers' and props.get('markerstyle'):
            size = shape.pop('size', 6)
            symbol = shape.pop('symbol')
            x0 = 0
            y0 = 0
            x1 = size
            y1 = size
            markerpath = props['markerstyle'].get('markerpath')
            if markerpath is None and symbol != 'circle':
                self.msg += 'not sure how to handle this marker without a valid path\n'
                return
            path = ' '.join([f'{a} {t[0]},{t[1]}' for a, t in zip(markerpath[1], markerpath[0])])
            if symbol == 'circle':
                path = None
                shape_type = 'circle'
                x0 = -size / 2
                y0 = size / 2
                x1 = size / 2
                y1 = size + size / 2
            else:
                shape_type = 'path'
            legend_shape = go.layout.Shape(type=shape_type, xref='paper', yref='paper', x0=x0, y0=y0, x1=x1, y1=y1, xsizemode='pixel', ysizemode='pixel', xanchor=x, yanchor=y, path=path, **shape)
        elif single_mode == 'lines':
            mode = 'line'
            x1 = props['data'][1][0]
            y1 = props['data'][1][1]
            legend_shape = go.layout.Shape(type=mode, xref='paper', yref='paper', x0=x, y0=y + 0.02, x1=x1, y1=y1 + 0.02, **shape)
        else:
            self.msg += 'not sure how to handle this element\n'
            return
        self.plotly_fig.add_shape(legend_shape)
        self.msg += '    Heck yeah, I drew that shape\n'