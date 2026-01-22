import gast as ast
from pythran.analyses import OrderedGlobalDeclarations
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.types.type_dependencies import TypeDependencies
import pythran.graph as graph

        Keep everything but function definition then add sorted functions.

        Most of the time, many function sort work so we use function calldepth
        as a "sort hint" to simplify typing.
        