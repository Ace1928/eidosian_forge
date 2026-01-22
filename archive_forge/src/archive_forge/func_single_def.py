from pythran.analyses import UseDefChains, Ancestors, Aliases, RangeValues
from pythran.analyses import Identifiers
from pythran.passmanager import Transformation
from pythran.tables import MODULES
import gast as ast
from copy import deepcopy
def single_def(self, node):
    chain = self.use_def_chains[node]
    return len(chain) == 1 and chain[0].node