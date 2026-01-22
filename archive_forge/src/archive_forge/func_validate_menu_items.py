from __future__ import annotations
import random
from textwrap import dedent
from typing import TYPE_CHECKING, Final, Literal, Mapping, Union, cast
from typing_extensions import TypeAlias
from streamlit.elements import image
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg as ForwardProto
from streamlit.proto.PageConfig_pb2 import PageConfig as PageConfigProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.string_util import is_emoji
from streamlit.url_util import is_url
from streamlit.util import lower_clean_dict_keys
def validate_menu_items(menu_items: MenuItems) -> None:
    for k, v in menu_items.items():
        if not valid_menu_item_key(k):
            raise StreamlitAPIException(f'We only accept the keys: "Get help", "Report a bug", and "About" ("{k}" is not a valid key.)')
        if v is not None and (not is_url(v, ('http', 'https', 'mailto')) and k != ABOUT_KEY):
            raise StreamlitAPIException(f'"{v}" is a not a valid URL!')