import decimal
from .draft04 import CodeGeneratorDraft04, JSON_TYPE_TO_PYTHON_TYPE
from .exceptions import JsonSchemaDefinitionException
from .generator import enforce_list
def generate_boolean_schema(self):
    """
        Means that schema can be specified by boolean.
        True means everything is valid, False everything is invalid.
        """
    if self._definition is True:
        self.l('pass')
    if self._definition is False:
        self.exc('{name} must not be there')