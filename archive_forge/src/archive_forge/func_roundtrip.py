import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def roundtrip(filename, output=sys.stdout):
    with open(filename, 'r') as pyfile:
        source = pyfile.read()
    tree = compile(source, filename, 'exec', ast.PyCF_ONLY_AST)
    Unparser(tree, output)