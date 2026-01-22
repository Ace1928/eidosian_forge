from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def get_decrease(self):
    """
        Flatten decrease data and get decrease text

        :rtype (list, list, list): flat_decrease_x: x-values for the decreasing
            trace, flat_decrease_y: y=values for the decreasing trace and
            text_decrease: hovertext for the decreasing trace
        """
    flat_decrease_x = utils.flatten(self.decrease_x)
    flat_decrease_y = utils.flatten(self.decrease_y)
    text_decrease = ('Open', 'Open', 'High', 'Low', 'Close', 'Close', '') * len(self.decrease_x)
    return (flat_decrease_x, flat_decrease_y, text_decrease)