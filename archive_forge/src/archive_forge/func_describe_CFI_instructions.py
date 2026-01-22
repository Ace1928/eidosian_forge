from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def describe_CFI_instructions(entry):
    """ Given a CFI entry (CIE or FDE), return the textual description of its
        instructions.
    """

    def _assert_FDE_instruction(instr):
        dwarf_assert(isinstance(entry, FDE), 'Unexpected instruction "%s" for a CIE' % instr)

    def _full_reg_name(regnum):
        regname = describe_reg_name(regnum, _MACHINE_ARCH, False)
        if regname:
            return 'r%s (%s)' % (regnum, regname)
        else:
            return 'r%s' % regnum
    if isinstance(entry, CIE):
        cie = entry
    else:
        cie = entry.cie
        pc = entry['initial_location']
    s = ''
    for instr in entry.instructions:
        name = instruction_name(instr.opcode)
        if name in ('DW_CFA_offset', 'DW_CFA_offset_extended', 'DW_CFA_offset_extended_sf', 'DW_CFA_val_offset', 'DW_CFA_val_offset_sf'):
            s += '  %s: %s at cfa%+d\n' % (name, _full_reg_name(instr.args[0]), instr.args[1] * cie['data_alignment_factor'])
        elif name in ('DW_CFA_restore', 'DW_CFA_restore_extended', 'DW_CFA_undefined', 'DW_CFA_same_value', 'DW_CFA_def_cfa_register'):
            s += '  %s: %s\n' % (name, _full_reg_name(instr.args[0]))
        elif name == 'DW_CFA_register':
            s += '  %s: %s in %s' % (name, _full_reg_name(instr.args[0]), _full_reg_name(instr.args[1]))
        elif name == 'DW_CFA_set_loc':
            pc = instr.args[0]
            s += '  %s: %08x\n' % (name, pc)
        elif name in ('DW_CFA_advance_loc1', 'DW_CFA_advance_loc2', 'DW_CFA_advance_loc4', 'DW_CFA_advance_loc'):
            _assert_FDE_instruction(instr)
            factored_offset = instr.args[0] * cie['code_alignment_factor']
            s += '  %s: %s to %08x\n' % (name, factored_offset, factored_offset + pc)
            pc += factored_offset
        elif name in ('DW_CFA_remember_state', 'DW_CFA_restore_state', 'DW_CFA_nop', 'DW_CFA_AARCH64_negate_ra_state'):
            s += '  %s\n' % name
        elif name == 'DW_CFA_def_cfa':
            s += '  %s: %s ofs %s\n' % (name, _full_reg_name(instr.args[0]), instr.args[1])
        elif name == 'DW_CFA_def_cfa_sf':
            s += '  %s: %s ofs %s\n' % (name, _full_reg_name(instr.args[0]), instr.args[1] * cie['data_alignment_factor'])
        elif name in ('DW_CFA_def_cfa_offset', 'DW_CFA_GNU_args_size'):
            s += '  %s: %s\n' % (name, instr.args[0])
        elif name == 'DW_CFA_def_cfa_expression':
            expr_dumper = ExprDumper(entry.structs)
            s += '  %s (%s)\n' % (name, expr_dumper.dump_expr(instr.args[0]))
        elif name == 'DW_CFA_expression':
            expr_dumper = ExprDumper(entry.structs)
            s += '  %s: %s (%s)\n' % (name, _full_reg_name(instr.args[0]), expr_dumper.dump_expr(instr.args[1]))
        else:
            s += '  %s: <??>\n' % name
    return s