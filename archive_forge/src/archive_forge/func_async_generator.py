import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
def async_generator(coroutine_maker):

    @wraps(coroutine_maker)
    def async_generator_maker(*args, **kwargs):
        return AsyncGenerator(coroutine_maker(*args, **kwargs))
    async_generator_maker._async_gen_function = id(async_generator_maker)
    return async_generator_maker