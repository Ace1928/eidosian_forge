import collections
from absl import logging
def get_fn(option):
    if name not in option._options:
        option._options[name] = default_factory()
    return option._options.get(name)