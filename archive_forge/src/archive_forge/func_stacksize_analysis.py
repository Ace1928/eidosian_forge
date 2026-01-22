import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
def stacksize_analysis(instructions) -> Union[int, float]:
    assert instructions
    fixed_point = FixedPointBox()
    stack_sizes = {inst: StackSize(float('inf'), float('-inf'), fixed_point) for inst in instructions}
    stack_sizes[instructions[0]].zero()
    for _ in range(100):
        if fixed_point.value:
            break
        fixed_point.value = True
        for inst, next_inst in zip(instructions, instructions[1:] + [None]):
            stack_size = stack_sizes[inst]
            is_call_finally = sys.version_info < (3, 9) and inst.opcode == dis.opmap['CALL_FINALLY']
            if inst.opcode not in TERMINAL_OPCODES:
                assert next_inst is not None, f'missing next inst: {inst}'
                stack_sizes[next_inst].offset_of(stack_size, stack_effect(inst.opcode, inst.arg, jump=is_call_finally))
            if inst.opcode in JUMP_OPCODES and (not is_call_finally):
                stack_sizes[inst.target].offset_of(stack_size, stack_effect(inst.opcode, inst.arg, jump=True))
            if inst.exn_tab_entry:
                depth = inst.exn_tab_entry.depth + int(inst.exn_tab_entry.lasti) + 1
                stack_sizes[inst.exn_tab_entry.target].exn_tab_jump(depth)
    if False:
        for inst in instructions:
            stack_size = stack_sizes[inst]
            print(stack_size.low, stack_size.high, inst)
    low = min([x.low for x in stack_sizes.values()])
    high = max([x.high for x in stack_sizes.values()])
    assert fixed_point.value, 'failed to reach fixed point'
    assert low >= 0
    return high