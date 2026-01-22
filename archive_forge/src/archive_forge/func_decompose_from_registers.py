from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import reflection_using_prepare as rup
from cirq_ft.algos import select_and_prepare as sp
from cirq_ft.algos.mean_estimation import complex_phase_oracle
def decompose_from_registers(self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
    select_reg = {reg.name: quregs[reg.name] for reg in self.select.signature}
    reflect_reg = {reg.name: quregs[reg.name] for reg in self.reflect.signature}
    select_op = self.select.on_registers(**select_reg)
    reflect_op = self.reflect.on_registers(**reflect_reg)
    for _ in range(self.power):
        yield select_op
        yield [reflect_op, cirq.global_phase_operation(-1)]