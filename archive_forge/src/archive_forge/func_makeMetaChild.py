from _paramtreecfg import cfg
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.Parameter import PARAM_TYPES
from pyqtgraph.parametertree.parameterTypes import GroupParameter
def makeMetaChild(name, cfgDict):
    children = []
    for chName, chOpts in cfgDict.items():
        if not isinstance(chOpts, dict):
            ch = Parameter.create(name=chName, type=chName, value=chOpts)
        else:
            ch = Parameter.create(name=chName, **chOpts)
        _encounteredTypes.add(ch.type())
        children.append(ch)
    param = Parameter.create(name=name, type='group', children=children)
    param.setOpts(expanded=False)
    return param