import matplotlib.axes as maxes
from matplotlib.artist import Artist
from matplotlib.axis import XAxis, YAxis
@property
def major_ticks(self):
    tickline = 'tick%dline' % self._axisnum
    return SimpleChainedObjects([getattr(tick, tickline) for tick in self._axis.get_major_ticks()])