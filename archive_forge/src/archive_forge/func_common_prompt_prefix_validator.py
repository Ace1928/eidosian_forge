from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def common_prompt_prefix_validator(df: pd.DataFrame) -> Remediation:
    """
    This validator will suggest to remove a common prefix from the prompt if a long one exist.
    """
    MAX_PREFIX_LEN = 12
    immediate_msg = None
    optional_msg = None
    optional_fn = None
    common_prefix = get_common_xfix(df.prompt, xfix='prefix')
    if common_prefix == '':
        return Remediation(name='common_prefix')

    def remove_common_prefix(x: Any, prefix: Any) -> Any:
        x['prompt'] = x['prompt'].str[len(prefix):]
        return x
    if (df.prompt == common_prefix).all():
        return Remediation(name='common_prefix')
    if common_prefix != '':
        immediate_msg = f'\n- All prompts start with prefix `{common_prefix}`'
        if MAX_PREFIX_LEN < len(common_prefix):
            immediate_msg += ". Fine-tuning doesn't require the instruction specifying the task, or a few-shot example scenario. Most of the time you should only add the input data into the prompt, and the desired output into the completion"
            optional_msg = f'Remove prefix `{common_prefix}` from all prompts'

            def optional_fn(x: Any) -> Any:
                return remove_common_prefix(x, common_prefix)
    return Remediation(name='common_prompt_prefix', immediate_msg=immediate_msg, optional_msg=optional_msg, optional_fn=optional_fn)