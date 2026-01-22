from pythran.analyses import (ImportedIds, HasReturn, IsAssigned, CFG,
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy

        generic expression splitting algorithm. Should work for ifexp and if
        using W(rap) and U(n)W(rap) to manage difference between expr and stmt

        The idea is to split a BinOp in three expressions:
            1. a (possibly empty) non-static expr
            2. an expr containing a static expr
            3. a (possibly empty) non-static expr
        Once split, the if body is refactored to keep the semantic,
        and then recursively split again, until all static expr are alone in a
        test condition
        