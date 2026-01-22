import json
import pathlib
import jsonschema
def make_validator(key):
    """make a JSON Schema (Draft 7) validator"""
    schema = {'$ref': '#/definitions/{}'.format(key)}
    schema.update(SCHEMA)
    return jsonschema.validators.Draft7Validator(schema)