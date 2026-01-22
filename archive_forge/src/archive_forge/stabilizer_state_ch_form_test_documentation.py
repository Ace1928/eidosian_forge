import numpy as np
import pytest
import cirq
import cirq.testing

    0: ───H───@───────────────X───M───────────
              │
    1: ───────X───@───────X───────────X───M───
                  │                   │
    2: ───────────X───M───────────────@───────

    After the third moment, before the measurement, the state is |000> + |111>.
    After measurement of q2, q0 and q1 both get a bit flip, so the q0
    measurement always yields opposite of the q2 measurement. q1 has an
    additional controlled not from q2, making it yield 1 always when measured.
    If there were no measurements in the circuit, the final state would be
    |110> + |011>.
    