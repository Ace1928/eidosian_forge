import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
def test_service_get_current_calibration():
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    calibration_dict = {'qubits': 11}
    mock_client.get_current_calibration.return_value = calibration_dict
    service._client = mock_client
    cal = service.get_current_calibration()
    assert cal.num_qubits() == 11
    mock_client.get_current_calibration.assert_called_once()