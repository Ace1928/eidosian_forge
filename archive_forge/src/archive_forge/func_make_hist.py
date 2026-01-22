from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def make_hist(self):
    """
        Makes the histogram(s) for FigureFactory.create_distplot().

        :rtype (list) hist: list of histogram representations
        """
    hist = [None] * self.trace_number
    for index in range(self.trace_number):
        hist[index] = dict(type='histogram', x=self.hist_data[index], xaxis='x1', yaxis='y1', histnorm=self.histnorm, name=self.group_labels[index], legendgroup=self.group_labels[index], marker=dict(color=self.colors[index % len(self.colors)]), autobinx=False, xbins=dict(start=self.start[index], end=self.end[index], size=self.bin_size[index]), opacity=0.7)
    return hist