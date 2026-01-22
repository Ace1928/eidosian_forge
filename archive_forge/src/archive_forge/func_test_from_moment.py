import os
from typing import cast
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import sympy
from google.protobuf import text_format
import cirq
import cirq_google
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.calibration.phased_fsim import (
from cirq_google.serialization.arg_func_langs import arg_to_proto
def test_from_moment():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    m = cirq.Moment(cirq.ISWAP(q_00, q_01) ** 0.5, cirq.ISWAP(q_02, q_03) ** 0.5)
    options = FloquetPhasedFSimCalibrationOptions(characterize_theta=True, characterize_zeta=True, characterize_chi=False, characterize_gamma=False, characterize_phi=True)
    request = FloquetPhasedFSimCalibrationRequest.from_moment(m, options)
    assert request == FloquetPhasedFSimCalibrationRequest(gate=cirq.ISWAP ** 0.5, pairs=((q_00, q_01), (q_02, q_03)), options=options)
    non_identical = cirq.Moment(cirq.ISWAP(q_00, q_01) ** 0.5, cirq.ISWAP(q_02, q_03))
    with pytest.raises(ValueError, match='must be identical'):
        _ = FloquetPhasedFSimCalibrationRequest.from_moment(non_identical, options)
    sq = cirq.Moment(cirq.X(q_00))
    with pytest.raises(ValueError, match='must be two qubit gates'):
        _ = FloquetPhasedFSimCalibrationRequest.from_moment(sq, options)
    threeq = cirq.Moment(cirq.TOFFOLI(q_00, q_01, q_02))
    with pytest.raises(ValueError, match='must be two qubit gates'):
        _ = FloquetPhasedFSimCalibrationRequest.from_moment(threeq, options)
    not_gate = cirq.Moment(cirq.CircuitOperation(cirq.FrozenCircuit()))
    with pytest.raises(ValueError, match='must be two qubit gates'):
        _ = FloquetPhasedFSimCalibrationRequest.from_moment(not_gate, options)
    empty = cirq.Moment()
    with pytest.raises(ValueError, match='No gates found'):
        _ = FloquetPhasedFSimCalibrationRequest.from_moment(empty, options)