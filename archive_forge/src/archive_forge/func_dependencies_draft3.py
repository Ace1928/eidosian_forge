import re
from referencing.jsonschema import lookup_recursive_ref
from jsonschema import _utils
from jsonschema.exceptions import ValidationError
def dependencies_draft3(validator, dependencies, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    for property, dependency in dependencies.items():
        if property not in instance:
            continue
        if validator.is_type(dependency, 'object'):
            yield from validator.descend(instance, dependency, schema_path=property)
        elif validator.is_type(dependency, 'string'):
            if dependency not in instance:
                message = f'{dependency!r} is a dependency of {property!r}'
                yield ValidationError(message)
        else:
            for each in dependency:
                if each not in instance:
                    message = f'{each!r} is a dependency of {property!r}'
                    yield ValidationError(message)