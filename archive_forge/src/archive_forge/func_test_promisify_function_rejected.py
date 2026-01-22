from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_promisify_function_rejected(resolve):
    promisified_func = promisify(sum_function)
    result = promisified_func(None, None)
    assert isinstance(result, Promise)
    with raises(Exception) as exc_info_promise:
        result.get()
    with raises(Exception) as exc_info:
        sum_function(None, None)
    assert str(exc_info_promise.value) == str(exc_info.value)