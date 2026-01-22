from asyncio import coroutine
from pytest import mark
from time import sleep
from promise import Promise
@coroutine
def my_coro():
    yield from Promise.resolve(True)