from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
def is_cf_contig(specs):
    is_c_contig = is_f_contig = False
    if len(specs) == 1 and specs == [('direct', 'contig')]:
        is_c_contig = True
    elif specs[-1] == ('direct', 'contig') and all((axis == ('direct', 'follow') for axis in specs[:-1])):
        is_c_contig = True
    elif len(specs) > 1 and specs[0] == ('direct', 'contig') and all((axis == ('direct', 'follow') for axis in specs[1:])):
        is_f_contig = True
    return (is_c_contig, is_f_contig)