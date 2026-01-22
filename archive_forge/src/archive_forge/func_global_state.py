import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
@property
def global_state(self):
    res = super().global_state
    res['custom_formats'] = self._custom_formats
    return res