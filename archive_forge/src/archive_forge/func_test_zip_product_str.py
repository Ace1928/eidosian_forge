import pytest
import sympy
import cirq
def test_zip_product_str():
    assert str(cirq.UnitSweep + cirq.UnitSweep + cirq.UnitSweep) == 'cirq.UnitSweep + cirq.UnitSweep + cirq.UnitSweep'
    assert str(cirq.UnitSweep * cirq.UnitSweep * cirq.UnitSweep) == 'cirq.UnitSweep * cirq.UnitSweep * cirq.UnitSweep'
    assert str(cirq.UnitSweep + cirq.UnitSweep * cirq.UnitSweep) == 'cirq.UnitSweep + cirq.UnitSweep * cirq.UnitSweep'
    assert str((cirq.UnitSweep + cirq.UnitSweep) * cirq.UnitSweep) == '(cirq.UnitSweep + cirq.UnitSweep) * cirq.UnitSweep'