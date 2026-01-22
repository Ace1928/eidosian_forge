import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def peep_hole_list_to_tuple(func_ir):
    """
    This peephole rewrites a bytecode sequence new to Python 3.9 that looks
    like e.g.:

    def foo(a):
        return (*a,)

    41          0 BUILD_LIST               0
                2 LOAD_FAST                0 (a)
                4 LIST_EXTEND              1
                6 LIST_TO_TUPLE
                8 RETURN_VAL

    essentially, the unpacking of tuples is written as a list which is appended
    to/extended and then "magicked" into a tuple by the new LIST_TO_TUPLE
    opcode.

    This peephole repeatedly analyses the bytecode in a block looking for a
    window between a `LIST_TO_TUPLE` and `BUILD_LIST` and...

    1. Turns the BUILD_LIST into a BUILD_TUPLE
    2. Sets an accumulator's initial value as the target of the BUILD_TUPLE
    3. Searches for 'extend' on the original list and turns these into binary
       additions on the accumulator.
    4. Searches for 'append' on the original list and turns these into a
       `BUILD_TUPLE` which is then appended via binary addition to the
       accumulator.
    5. Assigns the accumulator to the variable that exits the peephole and the
       rest of the block/code refers to as the result of the unpack operation.
    6. Patches up
    """
    _DEBUG = False
    for offset, blk in func_ir.blocks.items():
        while True:

            def find_postive_region():
                found = False
                for idx in reversed(range(len(blk.body))):
                    stmt = blk.body[idx]
                    if isinstance(stmt, ir.Assign):
                        value = stmt.value
                        if isinstance(value, ir.Expr) and value.op == 'list_to_tuple':
                            target_list = value.info[0]
                            found = True
                            bt = (idx, stmt)
                    if found:
                        if isinstance(stmt, ir.Assign):
                            if stmt.target.name == target_list:
                                region = (bt, (idx, stmt))
                                return region
            region = find_postive_region()
            if region is not None:
                peep_hole = blk.body[region[1][0]:region[0][0]]
                if _DEBUG:
                    print('\nWINDOW:')
                    for x in peep_hole:
                        print(x)
                    print('')
                appends = []
                extends = []
                init = region[1][1]
                const_list = init.target.name
                for x in peep_hole:
                    if isinstance(x, ir.Assign):
                        if isinstance(x.value, ir.Expr):
                            expr = x.value
                            if expr.op == 'getattr' and expr.value.name == const_list:
                                if expr.attr == 'extend':
                                    extends.append(x.target.name)
                                elif expr.attr == 'append':
                                    appends.append(x.target.name)
                                else:
                                    assert 0
                new_hole = []

                def append_and_fix(x):
                    """ Adds to the new_hole and fixes up definitions"""
                    new_hole.append(x)
                    if x.target.name in func_ir._definitions:
                        assert len(func_ir._definitions[x.target.name]) == 1
                        func_ir._definitions[x.target.name].clear()
                    func_ir._definitions[x.target.name].append(x.value)
                the_build_list = init.target
                if _DEBUG:
                    print('\nBLOCK:')
                    blk.dump()
                t2l_agn = region[0][1]
                acc = the_build_list
                for x in peep_hole:
                    if isinstance(x, ir.Assign):
                        if isinstance(x.value, ir.Expr):
                            expr = x.value
                            if expr.op == 'getattr':
                                if x.target.name in extends or x.target.name in appends:
                                    func_ir._definitions.pop(x.target.name)
                                    continue
                                else:
                                    new_hole.append(x)
                            elif expr.op == 'call':
                                fname = expr.func.name
                                if fname in extends or fname in appends:
                                    arg = expr.args[0]
                                    if isinstance(arg, ir.Var):
                                        tmp_name = '%s_var_%s' % (fname, arg.name)
                                        if fname in appends:
                                            bt = ir.Expr.build_tuple([arg], expr.loc)
                                        else:
                                            gv_tuple = ir.Global(name='tuple', value=tuple, loc=expr.loc)
                                            tuple_var = arg.scope.redefine('$_list_extend_gv_tuple', loc=expr.loc)
                                            new_hole.append(ir.Assign(target=tuple_var, value=gv_tuple, loc=expr.loc))
                                            bt = ir.Expr.call(tuple_var, (arg,), (), loc=expr.loc)
                                        var = ir.Var(arg.scope, tmp_name, expr.loc)
                                        asgn = ir.Assign(bt, var, expr.loc)
                                        append_and_fix(asgn)
                                        arg = var
                                    new = ir.Expr.binop(fn=operator.add, lhs=acc, rhs=arg, loc=x.loc)
                                    asgn = ir.Assign(new, x.target, expr.loc)
                                    append_and_fix(asgn)
                                    acc = asgn.target
                                else:
                                    new_hole.append(x)
                            elif expr.op == 'build_list' and x.target.name == const_list:
                                new = ir.Expr.build_tuple(expr.items, expr.loc)
                                asgn = ir.Assign(new, x.target, expr.loc)
                                append_and_fix(asgn)
                            else:
                                new_hole.append(x)
                        else:
                            new_hole.append(x)
                    else:
                        new_hole.append(x)
                append_and_fix(ir.Assign(acc, t2l_agn.target, the_build_list.loc))
                if _DEBUG:
                    print('\nNEW HOLE:')
                    for x in new_hole:
                        print(x)
                cpy = blk.body[:]
                head = cpy[:region[1][0]]
                tail = blk.body[region[0][0] + 1:]
                tmp = head + new_hole + tail
                blk.body.clear()
                blk.body.extend(tmp)
                if _DEBUG:
                    print('\nDUMP post hole:')
                    blk.dump()
            else:
                break
    return func_ir