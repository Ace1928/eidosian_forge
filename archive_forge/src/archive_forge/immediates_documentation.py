from pythran.analyses import Aliases
from pythran.passmanager import NodeAnalysis
from pythran.utils import pythran_builtin, isnum, ispowi

Immediates gathers immediates. For now, only integers within shape are
and argument of functions flagged as immediate_arguments are
considered as immediates
