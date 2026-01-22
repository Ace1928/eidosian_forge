from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
@fixture(params=[Promise.for_dict, free_promise_for_dict])
def promise_for_dict(request):
    return request.param