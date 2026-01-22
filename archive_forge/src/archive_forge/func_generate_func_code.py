from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
def generate_func_code(self):
    """
        Creates base code of validation function and calls helper
        for creating code by definition.
        """
    self.l('NoneType = type(None)')
    while self._needed_validation_functions:
        uri, name = self._needed_validation_functions.popitem()
        self.generate_validation_function(uri, name)