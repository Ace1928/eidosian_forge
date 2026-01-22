import warnings
import plotly.graph_objs as go
from plotly.matplotlylib.mplexporter import Renderer
from plotly.matplotlylib import mpltools
def draw_title(self, **props):
    """Add a title to the current subplot in layout dictionary.

        If there exists more than a single plot in the figure, titles revert
        to 'page'-referenced annotations.

        props.keys() -- [
        'text',         (actual content string, not the text obj)
        'position',     (an x, y pair, not an mpl Bbox)
        'coordinates',  ('data', 'axes', 'figure', 'display')
        'text_type',    ('title', 'xlabel', or 'ylabel')
        'style',        (style dict, see below)
        'mplobj'        (actual mpl text object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of text)
        'fontsize',     (size in points of text)
        'color',        (hex color)
        'halign',       (horizontal alignment, 'left', 'center', or 'right')
        'valign',       (vertical alignment, 'baseline', 'center', or 'top')
        'rotation',
        'zorder',       (precedence of text when stacked with other objs)
        ]

        """
    self.msg += '        Attempting to draw a title\n'
    if len(self.mpl_fig.axes) > 1:
        self.msg += '          More than one subplot, adding title as annotation\n'
        x_px, y_px = props['mplobj'].get_transform().transform(props['position'])
        x, y = mpltools.display_to_paper(x_px, y_px, self.plotly_fig['layout'])
        annotation = go.layout.Annotation(text=props['text'], font=go.layout.annotation.Font(color=props['style']['color'], size=props['style']['fontsize']), xref='paper', yref='paper', x=x, y=y, xanchor='center', yanchor='bottom', showarrow=False)
        self.plotly_fig['layout']['annotations'] += (annotation,)
    else:
        self.msg += '          Only one subplot found, adding as a plotly title\n'
        self.plotly_fig['layout']['title'] = props['text']
        titlefont = dict(size=props['style']['fontsize'], color=props['style']['color'])
        self.plotly_fig['layout']['titlefont'] = titlefont