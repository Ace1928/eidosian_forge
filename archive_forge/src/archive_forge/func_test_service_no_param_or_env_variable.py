import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
@mock.patch.dict(os.environ, {}, clear=True)
def test_service_no_param_or_env_variable():
    with pytest.raises(EnvironmentError):
        _ = ionq.Service(remote_host='http://example.com')