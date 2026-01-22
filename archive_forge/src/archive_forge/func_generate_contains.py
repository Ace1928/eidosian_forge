import decimal
from .draft04 import CodeGeneratorDraft04, JSON_TYPE_TO_PYTHON_TYPE
from .exceptions import JsonSchemaDefinitionException
from .generator import enforce_list
def generate_contains(self):
    """
        Means that array must contain at least one defined item.

        .. code-block:: python

            {
                'contains': {
                    'type': 'number',
                },
            }

        Valid array is any with at least one number.
        """
    self.create_variable_is_list()
    with self.l('if {variable}_is_list:'):
        contains_definition = self._definition['contains']
        if contains_definition is False:
            self.exc('{name} is always invalid', rule='contains')
        elif contains_definition is True:
            with self.l('if not {variable}:'):
                self.exc('{name} must not be empty', rule='contains')
        else:
            self.l('{variable}_contains = False')
            with self.l('for {variable}_key in {variable}:'):
                with self.l('try:'):
                    self.generate_func_code_block(contains_definition, '{}_key'.format(self._variable), self._variable_name, clear_variables=True)
                    self.l('{variable}_contains = True')
                    self.l('break')
                self.l('except JsonSchemaValueException: pass')
            with self.l('if not {variable}_contains:'):
                self.exc('{name} must contain one of contains definition', rule='contains')