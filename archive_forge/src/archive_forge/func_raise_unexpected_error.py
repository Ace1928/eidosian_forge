import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@utils.retry(exception.VolumeDeviceNotFound)
def raise_unexpected_error():
    raise WrongException('wrong exception')