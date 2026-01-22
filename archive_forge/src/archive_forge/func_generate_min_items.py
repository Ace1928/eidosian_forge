import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_min_items(self):
    self.create_variable_is_list()
    with self.l('if {variable}_is_list:'):
        if not isinstance(self._definition['minItems'], int):
            raise JsonSchemaDefinitionException('minItems must be a number')
        self.create_variable_with_length()
        with self.l('if {variable}_len < {minItems}:'):
            self.exc('{name} must contain at least {minItems} items', rule='minItems')