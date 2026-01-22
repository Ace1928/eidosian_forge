import gast as ast
import itertools
import os
from pythran.analyses import GlobalDeclarations
from pythran.errors import PythranInternalError
from pythran.passmanager import ModuleAnalysis
from pythran.types.conversion import PYTYPE_TO_CTYPE_TABLE
from pythran.utils import get_variable
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.graph import DiGraph
 Exception may declare a new variable. 