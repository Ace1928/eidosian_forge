from pythran.analyses.ancestors import AncestorsWithBody
from pythran.analyses.use_def_chain import DefUseChains
from pythran.passmanager import FunctionAnalysis
from collections import defaultdict
import gast as ast

    Associate each variable declaration with the node that defines it

    Whenever possible, associate the variable declaration to an assignment,
    otherwise to a node that defines a bloc (e.g. a For)
    This takes OpenMP information into accounts!
    The result is a dictionary with nodes as key and set of names as values
    