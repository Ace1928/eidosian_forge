from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
def create_variable_is_list(self):
    """
        Append code for creating variable with bool if it's instance of list
        with a name ``{variable}_is_list``. Similar to `create_variable_with_length`.
        """
    variable_name = '{}_is_list'.format(self._variable)
    if variable_name in self._variables:
        return
    self._variables.add(variable_name)
    self.l('{variable}_is_list = isinstance({variable}, (list, tuple))')