from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def get_all_xy(self):
    """
        Zip data to create OHLC shape

        OHLC shape: low to high vertical bar with
        horizontal branches for open and close values.
        If dates were added, the smallest date difference is calculated and
        multiplied by .2 to get the length of the open and close branches.
        If no date data was provided, the x-axis is a list of integers and the
        length of the open and close branches is .2.
        """
    self.all_y = list(zip(self.open, self.open, self.high, self.low, self.close, self.close, self.empty))
    if self.dates is not None:
        date_dif = []
        for i in range(len(self.dates) - 1):
            date_dif.append(self.dates[i + 1] - self.dates[i])
        date_dif_min = min(date_dif) / 5
        self.all_x = [[x - date_dif_min, x, x, x, x, x + date_dif_min, None] for x in self.dates]
    else:
        self.all_x = [[x - 0.2, x, x, x, x, x + 0.2, None] for x in range(len(self.open))]