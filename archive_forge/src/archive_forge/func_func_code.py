from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
@property
def func_code(self):
    """
        Returns generated code of whole validation function as string.
        """
    self._generate_func_code()
    return '\n'.join(self._code)