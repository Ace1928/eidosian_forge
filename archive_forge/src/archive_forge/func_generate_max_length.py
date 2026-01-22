import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_max_length(self):
    with self.l('if isinstance({variable}, str):'):
        self.create_variable_with_length()
        if not isinstance(self._definition['maxLength'], int):
            raise JsonSchemaDefinitionException('maxLength must be a number')
        with self.l('if {variable}_len > {maxLength}:'):
            self.exc('{name} must be shorter than or equal to {maxLength} characters', rule='maxLength')