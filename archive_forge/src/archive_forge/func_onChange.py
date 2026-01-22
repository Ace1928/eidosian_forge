from _paramtreecfg import cfg
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.Parameter import PARAM_TYPES
from pyqtgraph.parametertree.parameterTypes import GroupParameter
def onChange(_param, _val):
    if _val == 'Use span':
        span = slider.opts.pop('span', None)
        slider.setOpts(span=span)
    else:
        limits = slider.opts.pop('limits', None)
        slider.setOpts(limits=limits)