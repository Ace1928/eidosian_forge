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
def expand_numpy_2_args(args, defaults=None):
    if numpy.__version__[0] < '2':
        if defaults is not None:
            return {'args': args, 'defaults': defaults}
        else:
            return {'args': args}
    else:
        return {}