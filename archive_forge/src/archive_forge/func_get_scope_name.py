import contextlib
import json
import re
from urllib import parse as urlparse
from urllib.parse import unquote
from .exceptions import JsonSchemaDefinitionException
def get_scope_name(self):
    """
        Get current scope and return it as a valid function name.
        """
    name = 'validate_' + unquote(self.resolution_scope).replace('~1', '_').replace('~0', '_').replace('"', '')
    name = re.sub('($[^a-zA-Z]|[^a-zA-Z0-9])', '_', name)
    name = name.lower().rstrip('_')
    return name