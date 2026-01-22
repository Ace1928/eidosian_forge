import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('p_x,p_y,p_z', ((-0.1, 0.0, 0.0), (0.0, -0.1, 0.0), (0.0, 0.0, -0.1), (0.1, -0.1, 0.1)))
def test_asymmetric_depolarizing_channel_negative_probability(p_x, p_y, p_z):
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.asymmetric_depolarize(p_x, p_y, p_z)