from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import reflection_using_prepare as rup
from cirq_ft.algos import select_and_prepare as sp
from cirq_ft.algos.mean_estimation import complex_phase_oracle
def controlled(self, num_controls: Optional[int]=None, control_values: Optional[Union[cirq.ops.AbstractControlValues, Sequence[Union[int, Collection[int]]]]]=None, control_qid_shape: Optional[Tuple[int, ...]]=None) -> 'MeanEstimationOperator':
    if num_controls is None:
        num_controls = 1
    if control_values is None:
        control_values = [1] * num_controls
    if isinstance(control_values, Sequence) and len(control_values) == 1 and isinstance(control_values[0], int) and (not self.cv):
        c_select = self.code.encoder.controlled(control_values=control_values)
        assert isinstance(c_select, sp.SelectOracle)
        return MeanEstimationOperator(CodeForRandomVariable(encoder=c_select, synthesizer=self.code.synthesizer), cv=self.cv + (control_values[0],), power=self.power, arctan_bitsize=self.arctan_bitsize)
    raise NotImplementedError(f'Cannot create a controlled version of {self} with control_values={control_values}.')