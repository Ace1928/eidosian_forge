import json as pyjson
from functools import singledispatch
from typing import Callable, Optional, Union
from pydantic import BaseModel
from outlines.fsm.json_schema import build_regex_from_schema, get_schema_from_signature
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial
from .regex import regex
@json.register(OpenAI)
def json_openai(model, schema_object: Union[str, object, Callable], sampler: Sampler=multinomial()):
    raise NotImplementedError('Cannot use JSON Schema-structure generation with an OpenAI model ' + 'due to the limitations of the OpenAI API')