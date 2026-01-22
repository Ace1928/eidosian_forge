from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def lower_case_validator(df: pd.DataFrame, column: Any) -> Remediation | None:
    """
    This validator will suggest to lowercase the column values, if more than a third of letters are uppercase.
    """

    def lower_case(x: Any) -> Any:
        x[column] = x[column].str.lower()
        return x
    count_upper = df[column].apply(lambda x: sum((1 for c in x if c.isalpha() and c.isupper()))).sum()
    count_lower = df[column].apply(lambda x: sum((1 for c in x if c.isalpha() and c.islower()))).sum()
    if count_upper * 2 > count_lower:
        return Remediation(name='lower_case', immediate_msg=f'\n- More than a third of your `{column}` column/key is uppercase. Uppercase {column}s tends to perform worse than a mixture of case encountered in normal language. We recommend to lower case the data if that makes sense in your domain. See https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset for more details', optional_msg=f'Lowercase all your data in column/key `{column}`', optional_fn=lower_case)
    return None