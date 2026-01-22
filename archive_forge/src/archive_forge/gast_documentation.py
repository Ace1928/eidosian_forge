import sys as _sys
import ast as _ast
from ast import boolop, cmpop, excepthandler, expr, expr_context, operator
from ast import slice, stmt, unaryop, mod, AST
from ast import iter_child_nodes, walk

    Increment the line number and end line number of each node in the tree
    starting at *node* by *n*. This is useful to "move code" to a different
    location in a file.
    