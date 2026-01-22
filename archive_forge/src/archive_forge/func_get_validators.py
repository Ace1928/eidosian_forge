from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def get_validators() -> list[Validator]:
    return [num_examples_validator, lambda x: necessary_column_validator(x, 'prompt'), lambda x: necessary_column_validator(x, 'completion'), additional_column_validator, non_empty_field_validator, format_inferrer_validator, duplicated_rows_validator, long_examples_validator, lambda x: lower_case_validator(x, 'prompt'), lambda x: lower_case_validator(x, 'completion'), common_prompt_suffix_validator, common_prompt_prefix_validator, common_completion_prefix_validator, common_completion_suffix_validator, completions_space_start_validator]