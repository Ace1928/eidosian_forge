import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_enum(self):
    """
        Means that only value specified in the enum is valid.

        .. code-block:: python

            {
                'enum': ['a', 'b'],
            }
        """
    enum = self._definition['enum']
    if not isinstance(enum, (list, tuple)):
        raise JsonSchemaDefinitionException('enum must be an array')
    with self.l('if {variable} not in {enum}:'):
        self.exc('{name} must be one of {}', self.e(enum), rule='enum')