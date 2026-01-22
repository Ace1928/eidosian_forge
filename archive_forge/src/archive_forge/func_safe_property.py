import functools
import re
import os
def safe_property(func):
    return property(reraise_uncaught(func))