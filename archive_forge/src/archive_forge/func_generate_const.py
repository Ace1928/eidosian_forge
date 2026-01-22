import decimal
from .draft04 import CodeGeneratorDraft04, JSON_TYPE_TO_PYTHON_TYPE
from .exceptions import JsonSchemaDefinitionException
from .generator import enforce_list
def generate_const(self):
    """
        Means that value is valid when is equeal to const definition.

        .. code-block:: python

            {
                'const': 42,
            }

        Only valid value is 42 in this example.
        """
    const = self._definition['const']
    if isinstance(const, str):
        const = '"{}"'.format(self.e(const))
    with self.l('if {variable} != {}:', const):
        self.exc('{name} must be same as const definition: {definition_rule}', rule='const')