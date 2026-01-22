import types
from typing_extensions import NoReturn
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def returnsFailure() -> Failure:
    try:
        raise SampleException()
    except SampleException:
        return Failure()