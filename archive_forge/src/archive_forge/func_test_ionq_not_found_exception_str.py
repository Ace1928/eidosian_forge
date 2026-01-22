import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
def test_ionq_not_found_exception_str():
    ex = ionq.IonQNotFoundException('err')
    assert str(ex) == "Status code: 404, Message: 'err'"