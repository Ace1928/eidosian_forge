from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_promisify_function_resolved(resolve):
    promisified_func = promisify(sum_function)
    result = promisified_func(1, 2)
    assert isinstance(result, Promise)
    assert result.get() == 3