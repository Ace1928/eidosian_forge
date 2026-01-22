import gast as ast
from importlib import import_module
import inspect
import logging
import numpy
import sys
from pythran.typing import Dict, Set, List, TypeVar, Union, Optional, NDArray
from pythran.typing import Generator, Fun, Tuple, Iterable, Sized, File
from pythran.conversion import to_ast, ToNotEval
from pythran.intrinsic import Class
from pythran.intrinsic import ClassWithConstConstructor, ExceptionClass
from pythran.intrinsic import ClassWithReadOnceConstructor
from pythran.intrinsic import ConstFunctionIntr, FunctionIntr, UpdateEffect
from pythran.intrinsic import ConstMethodIntr, MethodIntr
from pythran.intrinsic import AttributeIntr, StaticAttributeIntr
from pythran.intrinsic import ReadEffect, ConstantIntr, UFunc
from pythran.intrinsic import ReadOnceMethodIntr
from pythran.intrinsic import ReadOnceFunctionIntr, ConstExceptionIntr
from pythran import interval
import beniget
def update_effects(self, node):
    """
    Combiner when we update the first argument of a function.

    It turn type of first parameter in combination of all others
    parameters types.
    """
    return [self.combine(node.args[0], None, node_args_k) for node_args_k in node.args[1:]]