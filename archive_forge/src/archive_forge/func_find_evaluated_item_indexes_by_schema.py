from collections.abc import Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit
import itertools
import re
def find_evaluated_item_indexes_by_schema(validator, instance, schema):
    """
    Get all indexes of items that get evaluated under the current schema.

    Covers all keywords related to unevaluatedItems: items, prefixItems, if,
    then, else, contains, unevaluatedItems, allOf, oneOf, anyOf
    """
    if validator.is_type(schema, 'boolean'):
        return []
    evaluated_indexes = []
    if 'items' in schema:
        return list(range(len(instance)))
    ref = schema.get('$ref')
    if ref is not None:
        resolved = validator._resolver.lookup(ref)
        evaluated_indexes.extend(find_evaluated_item_indexes_by_schema(validator.evolve(schema=resolved.contents, _resolver=resolved.resolver), instance, resolved.contents))
    dynamicRef = schema.get('$dynamicRef')
    if dynamicRef is not None:
        resolved = validator._resolver.lookup(dynamicRef)
        evaluated_indexes.extend(find_evaluated_item_indexes_by_schema(validator.evolve(schema=resolved.contents, _resolver=resolved.resolver), instance, resolved.contents))
    if 'prefixItems' in schema:
        evaluated_indexes += list(range(len(schema['prefixItems'])))
    if 'if' in schema:
        if validator.evolve(schema=schema['if']).is_valid(instance):
            evaluated_indexes += find_evaluated_item_indexes_by_schema(validator, instance, schema['if'])
            if 'then' in schema:
                evaluated_indexes += find_evaluated_item_indexes_by_schema(validator, instance, schema['then'])
        elif 'else' in schema:
            evaluated_indexes += find_evaluated_item_indexes_by_schema(validator, instance, schema['else'])
    for keyword in ['contains', 'unevaluatedItems']:
        if keyword in schema:
            for k, v in enumerate(instance):
                if validator.evolve(schema=schema[keyword]).is_valid(v):
                    evaluated_indexes.append(k)
    for keyword in ['allOf', 'oneOf', 'anyOf']:
        if keyword in schema:
            for subschema in schema[keyword]:
                errs = next(validator.descend(instance, subschema), None)
                if errs is None:
                    evaluated_indexes += find_evaluated_item_indexes_by_schema(validator, instance, subschema)
    return evaluated_indexes