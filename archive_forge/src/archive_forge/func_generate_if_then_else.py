from .draft06 import CodeGeneratorDraft06
def generate_if_then_else(self):
    """
        Implementation of if-then-else.

        .. code-block:: python

            {
                'if': {
                    'exclusiveMaximum': 0,
                },
                'then': {
                    'minimum': -10,
                },
                'else': {
                    'multipleOf': 2,
                },
            }

        Valid values are any between -10 and 0 or any multiplication of two.
        """
    with self.l('try:', optimize=False):
        self.generate_func_code_block(self._definition['if'], self._variable, self._variable_name, clear_variables=True)
    with self.l('except JsonSchemaValueException:'):
        if 'else' in self._definition:
            self.generate_func_code_block(self._definition['else'], self._variable, self._variable_name, clear_variables=True)
        else:
            self.l('pass')
    if 'then' in self._definition:
        with self.l('else:'):
            self.generate_func_code_block(self._definition['then'], self._variable, self._variable_name, clear_variables=True)