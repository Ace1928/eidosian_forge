from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def make_rug(self):
    """
        Makes the rug plot(s) for create_distplot().

        :rtype (list) rug: list of rug plot representations
        """
    rug = [None] * self.trace_number
    for index in range(self.trace_number):
        rug[index] = dict(type='scatter', x=self.hist_data[index], y=[self.group_labels[index]] * len(self.hist_data[index]), xaxis='x1', yaxis='y2', mode='markers', name=self.group_labels[index], legendgroup=self.group_labels[index], showlegend=False if self.show_hist or self.show_curve else True, text=self.rug_text[index], marker=dict(color=self.colors[index % len(self.colors)], symbol='line-ns-open'))
    return rug