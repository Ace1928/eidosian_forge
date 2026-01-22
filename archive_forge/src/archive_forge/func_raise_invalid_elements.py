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
def raise_invalid_elements(self, invalid_els):
    if invalid_els:
        raise ValueError("\n    Invalid element(s) received for the '{name}' property of {pname}\n        Invalid elements include: {invalid}\n\n{valid_clr_desc}".format(name=self.plotly_name, pname=self.parent_name, invalid=invalid_els[:10], valid_clr_desc=self.description()))