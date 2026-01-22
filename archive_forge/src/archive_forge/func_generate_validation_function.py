from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
def generate_validation_function(self, uri, name):
    """
        Generate validation function for given uri with given name
        """
    self._validation_functions_done.add(uri)
    self.l('')
    with self._resolver.resolving(uri) as definition:
        with self.l('def {}(data, custom_formats={{}}, name_prefix=None):', name):
            self.generate_func_code_block(definition, 'data', 'data', clear_variables=True)
            self.l('return data')