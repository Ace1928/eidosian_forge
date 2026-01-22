import sys
from copy import deepcopy
from functools import partial
from operator import mul, truediv
def setValues(self, values):
    assert len(values) == len(self.weights), 'Assigned values have not the same length than fitness weights'
    try:
        self.wvalues = tuple(map(mul, values, self.weights))
    except TypeError:
        _, _, traceback = sys.exc_info()
        raise TypeError('Both weights and assigned values must be a sequence of numbers when assigning to values of %r. Currently assigning value(s) %r of %r to a fitness with weights %s.' % (self.__class__, values, type(values), self.weights)).with_traceback(traceback)