from time import perf_counter
from .. import functions as fn
from ..Qt import QtWidgets
def averageValue(self):
    return sum((v[1] for v in self.values)) / float(len(self.values))