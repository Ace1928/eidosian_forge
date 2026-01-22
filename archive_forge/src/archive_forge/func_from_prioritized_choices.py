from __future__ import annotations as _annotations
import dataclasses
import inspect
import math
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import is_dataclass
from enum import Enum
from typing import (
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import Annotated, Literal, TypeAlias, assert_never, deprecated, final
from pydantic.warnings import PydanticDeprecatedSince26
from ._internal import (
from .annotated_handlers import GetJsonSchemaHandler
from .config import JsonDict, JsonSchemaExtraCallable, JsonValue
from .errors import PydanticInvalidForJsonSchema, PydanticUserError
@staticmethod
def from_prioritized_choices(prioritized_choices: dict[DefsRef, list[DefsRef]], defs_to_json: dict[DefsRef, JsonRef], definitions: dict[DefsRef, JsonSchemaValue]) -> _DefinitionsRemapping:
    """
        This function should produce a remapping that replaces complex DefsRef with the simpler ones from the
        prioritized_choices such that applying the name remapping would result in an equivalent JSON schema.
        """
    copied_definitions = deepcopy(definitions)
    definitions_schema = {'$defs': copied_definitions}
    for _iter in range(100):
        schemas_for_alternatives: dict[DefsRef, list[JsonSchemaValue]] = defaultdict(list)
        for defs_ref in copied_definitions:
            alternatives = prioritized_choices[defs_ref]
            for alternative in alternatives:
                schemas_for_alternatives[alternative].append(copied_definitions[defs_ref])
        for defs_ref, schemas in schemas_for_alternatives.items():
            schemas_for_alternatives[defs_ref] = _deduplicate_schemas(schemas_for_alternatives[defs_ref])
        defs_remapping: dict[DefsRef, DefsRef] = {}
        json_remapping: dict[JsonRef, JsonRef] = {}
        for original_defs_ref in definitions:
            alternatives = prioritized_choices[original_defs_ref]
            remapped_defs_ref = next((x for x in alternatives if len(schemas_for_alternatives[x]) == 1))
            defs_remapping[original_defs_ref] = remapped_defs_ref
            json_remapping[defs_to_json[original_defs_ref]] = defs_to_json[remapped_defs_ref]
        remapping = _DefinitionsRemapping(defs_remapping, json_remapping)
        new_definitions_schema = remapping.remap_json_schema({'$defs': copied_definitions})
        if definitions_schema == new_definitions_schema:
            return remapping
        definitions_schema = new_definitions_schema
    raise PydanticInvalidForJsonSchema('Failed to simplify the JSON schema definitions')