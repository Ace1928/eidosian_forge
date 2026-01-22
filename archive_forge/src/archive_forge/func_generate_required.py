import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_required(self):
    self.create_variable_is_dict()
    with self.l('if {variable}_is_dict:'):
        if not isinstance(self._definition['required'], (list, tuple)):
            raise JsonSchemaDefinitionException('required must be an array')
        self.l('{variable}__missing_keys = set({required}) - {variable}.keys()')
        with self.l('if {variable}__missing_keys:'):
            dynamic = 'str(sorted({variable}__missing_keys)) + " properties"'
            self.exc('{name} must contain ', self.e(self._definition['required']), rule='required', append_to_msg=dynamic)