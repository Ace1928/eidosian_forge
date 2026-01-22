from asyncio import coroutine
from pytest import mark
from time import sleep
from promise import Promise
@mark.asyncio
@coroutine
def test_await():
    yield from Promise.resolve(True)