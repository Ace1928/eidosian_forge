from __future__ import annotations
import os
import fastjsonschema
import jsonschema
from fastjsonschema import JsonSchemaException as _JsonSchemaException
from jsonschema import Draft4Validator as _JsonSchemaValidator
from jsonschema.exceptions import ErrorTree, ValidationError
def get_current_validator():
    """
    Return the default validator based on the value of an environment variable.
    """
    validator_name = os.environ.get('NBFORMAT_VALIDATOR', 'fastjsonschema')
    return _validator_for_name(validator_name)