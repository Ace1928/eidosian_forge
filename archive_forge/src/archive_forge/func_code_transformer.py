import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def code_transformer(self, code, context):
    if sys.flags.verbose:
        print('Optimize %s:%s: %s' % (code.co_filename, code.co_firstlineno, code.co_name))
    optimizer = PeepholeOptimizer()
    return optimizer.optimize(code)