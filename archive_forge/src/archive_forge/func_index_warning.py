from collections import Counter
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Constraint, Block
from pyomo.core.base.set import SetProduct
def index_warning(name, index):
    return 'WARNING: %s has no index %s' % (name, index)