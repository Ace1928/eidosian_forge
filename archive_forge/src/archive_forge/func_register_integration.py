import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
def register_integration(*toolkitnames):
    """Decorator to register an event loop to integrate with the IPython kernel

    The decorator takes names to register the event loop as for the %gui magic.
    You can provide alternative names for the same toolkit.

    The decorated function should take a single argument, the IPython kernel
    instance, arrange for the event loop to call ``kernel.do_one_iteration()``
    at least every ``kernel._poll_interval`` seconds, and start the event loop.

    :mod:`ipykernel.eventloops` provides and registers such functions
    for a few common event loops.
    """

    def decorator(func):
        """Integration registration decorator."""
        for name in toolkitnames:
            loop_map[name] = func
        func.exit_hook = lambda kernel: None

        def exit_decorator(exit_func):
            """@func.exit is now a decorator

            to register a function to be called on exit
            """
            func.exit_hook = exit_func
            return exit_func
        func.exit = exit_decorator
        return func
    return decorator