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
def validate_axes_specs(positions, specs, is_c_contig, is_f_contig):
    packing_specs = ('contig', 'strided', 'follow')
    access_specs = ('direct', 'ptr', 'full')
    has_contig = has_follow = has_strided = has_generic_contig = False
    last_indirect_dimension = -1
    for idx, (access, packing) in enumerate(specs):
        if access == 'ptr':
            last_indirect_dimension = idx
    for idx, (pos, (access, packing)) in enumerate(zip(positions, specs)):
        if not (access in access_specs and packing in packing_specs):
            raise CompileError(pos, 'Invalid axes specification.')
        if packing == 'strided':
            has_strided = True
        elif packing == 'contig':
            if has_contig:
                raise CompileError(pos, 'Only one direct contiguous axis may be specified.')
            valid_contig_dims = (last_indirect_dimension + 1, len(specs) - 1)
            if idx not in valid_contig_dims and access != 'ptr':
                if last_indirect_dimension + 1 != len(specs) - 1:
                    dims = 'dimensions %d and %d' % valid_contig_dims
                else:
                    dims = 'dimension %d' % valid_contig_dims[0]
                raise CompileError(pos, 'Only %s may be contiguous and direct' % dims)
            has_contig = access != 'ptr'
        elif packing == 'follow':
            if has_strided:
                raise CompileError(pos, 'A memoryview cannot have both follow and strided axis specifiers.')
            if not (is_c_contig or is_f_contig):
                raise CompileError(pos, 'Invalid use of the follow specifier.')
        if access in ('ptr', 'full'):
            has_strided = False