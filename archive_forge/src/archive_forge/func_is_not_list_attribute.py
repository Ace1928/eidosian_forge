import sys
import os
import re
import warnings
import types
import unicodedata
@classmethod
def is_not_list_attribute(cls, attr):
    """
        Returns True if and only if the given attribute is NOT one of the
        basic list attributes defined for all Elements.
        """
    return attr not in cls.list_attributes