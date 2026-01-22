from pythran.analyses import PotentialIterator, Aliases
from pythran.passmanager import Transformation
from pythran.utils import path_to_attr, path_to_node
Replace function call by its correct iterator if it is possible.