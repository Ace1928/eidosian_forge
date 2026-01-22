import logging
import os
def set_builtins_dir(path):
    """Sets the appropriate path for testing and reinitializes the module."""
    global _handler_dir, _available_builtins
    _handler_dir = path
    _available_builtins = []
    _initialize_builtins()