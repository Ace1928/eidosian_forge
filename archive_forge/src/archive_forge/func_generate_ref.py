from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
def generate_ref(self):
    """
        Ref can be link to remote or local definition.

        .. code-block:: python

            {'$ref': 'http://json-schema.org/draft-04/schema#'}
            {
                'properties': {
                    'foo': {'type': 'integer'},
                    'bar': {'$ref': '#/properties/foo'}
                }
            }
        """
    with self._resolver.in_scope(self._definition['$ref']):
        name = self._resolver.get_scope_name()
        uri = self._resolver.get_uri()
        if uri not in self._validation_functions_done:
            self._needed_validation_functions[uri] = name
        assert self._variable_name.startswith('data')
        path = self._variable_name[4:]
        name_arg = '(name_prefix or "data") + "{}"'.format(path)
        if '{' in name_arg:
            name_arg = name_arg + '.format(**locals())'
        self.l('{}({variable}, custom_formats, {name_arg})', name, name_arg=name_arg)