from pythran.analyses import ImportedIds
from pythran.passmanager import Transformation
from pythran.conversion import mangle
import pythran.metadata as metadata
import gast as ast
from functools import reduce

            Wrap comprehension content in all possibles if clauses.

            Examples
            --------
            >> [i for i in range(2) if i < 3 if 0 < i]

            Becomes

            >> for i in range(2):
            >>    if i < 3:
            >>        if 0 < i:
            >>            ... the code from `node` ...

            Note the nested ifs clauses.
            