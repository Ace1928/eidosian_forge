from pythran.passmanager import Transformation
from pythran.analyses import Ancestors
from pythran.syntax import PythranSyntaxError
from functools import reduce
import gast as ast
 NormalizeIsNone detects is None patterns. 