from oslo_utils import uuidutils
from urllib.parse import parse_qs
from urllib.parse import urlparse
from designateclient import exceptions
def resolve_by_name(func, name, *args):
    """
    Helper to resolve a "name" a'la foo.com to it's ID by using REST api's
    query support and filtering on name.
    """
    if uuidutils.is_uuid_like(name):
        return name
    results = func(*args, criterion={'name': f'{name}'})
    length = len(results)
    if length == 1:
        return results[0]['id']
    elif length == 0:
        raise exceptions.NotFound(f"Name {name} didn't resolve")
    else:
        raise exceptions.NoUniqueMatch('Multiple matches found for {name}, please use ID instead.')