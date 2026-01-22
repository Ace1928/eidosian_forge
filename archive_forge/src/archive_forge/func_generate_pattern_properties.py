import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_pattern_properties(self):
    """
        Means object with defined keys as patterns.

        .. code-block:: python

            {
                'patternProperties': {
                    '^x': {'type': 'number'},
                },
            }

        Valid object is containing key starting with a 'x' and value any number.
        """
    self.create_variable_is_dict()
    with self.l('if {variable}_is_dict:'):
        self.create_variable_keys()
        for pattern, definition in self._definition['patternProperties'].items():
            self._compile_regexps[pattern] = re.compile(pattern)
        with self.l('for {variable}_key, {variable}_val in {variable}.items():'):
            for pattern, definition in self._definition['patternProperties'].items():
                with self.l('if REGEX_PATTERNS[{}].search({variable}_key):', repr(pattern)):
                    with self.l('if {variable}_key in {variable}_keys:'):
                        self.l('{variable}_keys.remove({variable}_key)')
                    self.generate_func_code_block(definition, '{}_val'.format(self._variable), '{}.{{{}_key}}'.format(self._variable_name, self._variable), clear_variables=True)