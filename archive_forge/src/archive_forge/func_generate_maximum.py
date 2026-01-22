import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_maximum(self):
    with self.l('if isinstance({variable}, (int, float, Decimal)):'):
        if not isinstance(self._definition['maximum'], (int, float, decimal.Decimal)):
            raise JsonSchemaDefinitionException('maximum must be a number')
        if self._definition.get('exclusiveMaximum', False):
            with self.l('if {variable} >= {maximum}:'):
                self.exc('{name} must be smaller than {maximum}', rule='maximum')
        else:
            with self.l('if {variable} > {maximum}:'):
                self.exc('{name} must be smaller than or equal to {maximum}', rule='maximum')