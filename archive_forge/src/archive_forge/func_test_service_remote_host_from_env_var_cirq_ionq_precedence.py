import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
@mock.patch.dict(os.environ, {'CIRQ_IONQ_REMOTE_HOST': 'http://example.com', 'IONQ_REMOTE_HOST': 'not_this_host'})
def test_service_remote_host_from_env_var_cirq_ionq_precedence():
    service = ionq.Service(api_key='tomyheart')
    assert service.remote_host == 'http://example.com'