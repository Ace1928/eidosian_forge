import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
def test_ionq_exception_str():
    ex = ionq.IonQException('err', status_code=501)
    assert str(ex) == "Status code: 501, Message: 'err'"