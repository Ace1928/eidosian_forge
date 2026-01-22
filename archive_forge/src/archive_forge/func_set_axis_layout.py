from collections import OrderedDict
from plotly import exceptions, optional_imports
from plotly.graph_objs import graph_objs
def set_axis_layout(self, axis_key):
    """
        Sets and returns default axis object for dendrogram figure.

        :param (str) axis_key: E.g., 'xaxis', 'xaxis1', 'yaxis', yaxis1', etc.
        :rtype (dict): An axis_key dictionary with set parameters.

        """
    axis_defaults = {'type': 'linear', 'ticks': 'outside', 'mirror': 'allticks', 'rangemode': 'tozero', 'showticklabels': True, 'zeroline': False, 'showgrid': False, 'showline': True}
    if len(self.labels) != 0:
        axis_key_labels = self.xaxis
        if self.orientation in ['left', 'right']:
            axis_key_labels = self.yaxis
        if axis_key_labels not in self.layout:
            self.layout[axis_key_labels] = {}
        self.layout[axis_key_labels]['tickvals'] = [zv * self.sign[axis_key] for zv in self.zero_vals]
        self.layout[axis_key_labels]['ticktext'] = self.labels
        self.layout[axis_key_labels]['tickmode'] = 'array'
    self.layout[axis_key].update(axis_defaults)
    return self.layout[axis_key]