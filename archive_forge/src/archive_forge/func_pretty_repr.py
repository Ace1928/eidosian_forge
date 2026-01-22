from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts
from langchain_core.utils.interactive_env import is_interactive_env
def pretty_repr(self, html: bool=False) -> str:
    title = get_msg_title_repr(self.type.title() + ' Message', bold=html)
    if self.name is not None:
        title += f'\nName: {self.name}'
    return f'{title}\n\n{self.content}'