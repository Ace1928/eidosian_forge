import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_properties(self):
    """
        Means object with defined keys.

        .. code-block:: python

            {
                'properties': {
                    'key': {'type': 'number'},
                },
            }

        Valid object is containing key called 'key' and value any number.
        """
    self.create_variable_is_dict()
    with self.l('if {variable}_is_dict:'):
        self.create_variable_keys()
        for key, prop_definition in self._definition['properties'].items():
            key_name = re.sub('($[^a-zA-Z]|[^a-zA-Z0-9])', '', key)
            if not isinstance(prop_definition, (dict, bool)):
                raise JsonSchemaDefinitionException('{}[{}] must be object'.format(self._variable, key_name))
            with self.l('if "{}" in {variable}_keys:', self.e(key)):
                self.l('{variable}_keys.remove("{}")', self.e(key))
                self.l('{variable}__{0} = {variable}["{1}"]', key_name, self.e(key))
                self.generate_func_code_block(prop_definition, '{}__{}'.format(self._variable, key_name), '{}.{}'.format(self._variable_name, self.e(key)), clear_variables=True)
            if self._use_default and isinstance(prop_definition, dict) and ('default' in prop_definition):
                self.l('else: {variable}["{}"] = {}', self.e(key), repr(prop_definition['default']))