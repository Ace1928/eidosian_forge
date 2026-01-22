from . import version
import collections
from functools import wraps
import sys
import warnings
def inline_callbacks(func):
    """inline_callbacks helps you write Deferred-using code that looks like a
    regular sequential function. For example::

        def thingummy():
            thing = yield makeSomeRequestResultingInDeferred()
            print thing #the result! hoorj!
        thingummy = inline_callbacks(thingummy)

    When you call anything that results in a Deferred, you can simply yield it;
    your generator will automatically be resumed when the Deferred's result is
    available. The generator will be sent the result of the Deferred with the
    'send' method on generators, or if the result was a failure, 'throw'.

    Your inline_callbacks-enabled generator will return a Deferred object, which
    will result in the return value of the generator (or will fail with a
    failure object if your generator raises an unhandled exception). Note that
    you can't use return result to return a value; use return_value(result)
    instead. Falling off the end of the generator, or simply using return
    will cause the Deferred to have a result of None.

    The Deferred returned from your deferred generator may errback if your
    generator raised an exception::

        def thingummy():
            thing = yield makeSomeRequestResultingInDeferred()
            if thing == 'I love Twisted':
                # will become the result of the Deferred
                return_value('TWISTED IS GREAT!')
            else:
                # will trigger an errback
                raise Exception('DESTROY ALL LIFE')
        thingummy = inline_callbacks(thingummy)
    """

    @wraps(func)
    def unwind_generator(*args, **kwargs):
        return _inline_callbacks(None, func(*args, **kwargs), Deferred())
    return unwind_generator