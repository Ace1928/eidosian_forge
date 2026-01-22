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
def set_menu_items_proto(lowercase_menu_items, menu_items_proto) -> None:
    if GET_HELP_KEY in lowercase_menu_items:
        if lowercase_menu_items[GET_HELP_KEY] is not None:
            menu_items_proto.get_help_url = lowercase_menu_items[GET_HELP_KEY]
        else:
            menu_items_proto.hide_get_help = True
    if REPORT_A_BUG_KEY in lowercase_menu_items:
        if lowercase_menu_items[REPORT_A_BUG_KEY] is not None:
            menu_items_proto.report_a_bug_url = lowercase_menu_items[REPORT_A_BUG_KEY]
        else:
            menu_items_proto.hide_report_a_bug = True
    if ABOUT_KEY in lowercase_menu_items:
        if lowercase_menu_items[ABOUT_KEY] is not None:
            menu_items_proto.about_section_md = dedent(lowercase_menu_items[ABOUT_KEY])