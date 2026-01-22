from __future__ import annotations
import json
import pprint
import warnings
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional
from ._imports import import_item
from .corpus.words import generate_corpus_id
from .json_compat import ValidationError, _validator_for_name, get_current_validator
from .reader import get_version
from .warnings import DuplicateCellId, MissingIDFieldWarning
def better_validation_error(error, version, version_minor):
    """Get better ValidationError on oneOf failures

    oneOf errors aren't informative.
    if it's a cell type or output_type error,
    try validating directly based on the type for a better error message
    """
    if not len(error.schema_path):
        return error
    key = error.schema_path[-1]
    ref = None
    if key.endswith('Of'):
        if isinstance(error.instance, dict):
            if 'cell_type' in error.instance:
                ref = error.instance['cell_type'] + '_cell'
            elif 'output_type' in error.instance:
                ref = error.instance['output_type']
        if ref:
            try:
                validate(error.instance, ref, version=version, version_minor=version_minor)
            except ValidationError as sub_error:
                error.relative_path.extend(sub_error.relative_path)
                sub_error.relative_path = error.relative_path
                better = better_validation_error(sub_error, version, version_minor)
                if better.ref is None:
                    better.ref = ref
                return better
            except Exception:
                pass
    return NotebookValidationError(error, ref)