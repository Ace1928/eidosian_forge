import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
def find_invalid_els(self, orig, validated, invalid_els=None):
    """
        Helper method to find invalid elements in orig array.
        Elements are invalid if their corresponding element in
        the validated array is None.

        This method handles deeply nested list structures
        """
    if invalid_els is None:
        invalid_els = []
    for orig_el, validated_el in zip(orig, validated):
        if is_array(orig_el):
            self.find_invalid_els(orig_el, validated_el, invalid_els)
        elif validated_el is None:
            invalid_els.append(orig_el)
    return invalid_els