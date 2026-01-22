from typing import Callable, Optional, Sequence, Union
import cirq
def run_sweep(self, program: cirq.AbstractCircuit, params: cirq.Sweepable, repetitions: int=1) -> Sequence[cirq.Result]:
    self._validate_circuit([program], [params], repetitions)
    return self._sampler.run_sweep(program, params, repetitions)