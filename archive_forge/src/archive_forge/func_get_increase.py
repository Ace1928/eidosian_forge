from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def get_increase(self):
    """
        Flatten increase data and get increase text

        :rtype (list, list, list): flat_increase_x: x-values for the increasing
            trace, flat_increase_y: y=values for the increasing trace and
            text_increase: hovertext for the increasing trace
        """
    flat_increase_x = utils.flatten(self.increase_x)
    flat_increase_y = utils.flatten(self.increase_y)
    text_increase = ('Open', 'Open', 'High', 'Low', 'Close', 'Close', '') * len(self.increase_x)
    return (flat_increase_x, flat_increase_y, text_increase)