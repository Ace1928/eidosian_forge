from __future__ import annotations
import io
import os
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, BinaryIO, Final, Literal, TextIO, Union, cast
from typing_extensions import TypeAlias
from streamlit import runtime, source_util
from streamlit.elements.form import current_form_id, is_in_form
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.file_util import get_main_script_directory, normalize_path_join
from streamlit.proto.Button_pb2 import Button as ButtonProto
from streamlit.proto.DownloadButton_pb2 import DownloadButton as DownloadButtonProto
from streamlit.proto.LinkButton_pb2 import LinkButton as LinkButtonProto
from streamlit.proto.PageLink_pb2 import PageLink as PageLinkProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.string_util import validate_emoji
from streamlit.type_util import Key, to_key
@gather_metrics('page_link')
def page_link(self, page: str, *, label: str | None=None, icon: str | None=None, help: str | None=None, disabled: bool=False, use_container_width: bool | None=None) -> DeltaGenerator:
    """Display a link to another page in a multipage app or to an external page.

        If another page in a multipage app is specified, clicking ``st.page_link``
        stops the current page execution and runs the specified page as if the
        user clicked on it in the sidebar navigation.

        If an external page is specified, clicking ``st.page_link`` opens a new
        tab to the specified page. The current script run will continue if not
        complete.

        Parameters
        ----------
        page : str
            The file path (relative to the main script) of the page to switch to.
            Alternatively, this can be the URL to an external page (must start
            with "http://" or "https://").
        label : str
            The label for the page link. Labels are required for external pages.
            Labels can optionally contain Markdown and supports the following
            elements: Bold, Italics, Strikethroughs, Inline Code, and Emojis.

            This also supports:

            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.
              For a list of all supported codes,
              see https://share.streamlit.io/streamlit/emoji-shortcodes.

            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"
              must be on their own lines). Supported LaTeX functions are listed
              at https://katex.org/docs/supported.html.

            * Colored text, using the syntax ``:color[text to be colored]``,
              where ``color`` needs to be replaced with any of the following
              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.

            Unsupported elements are unwrapped so only their children (text contents)
            render. Display unsupported elements as literal characters by
            backslash-escaping them. E.g. ``1\\. Not an ordered list``.
        icon : str
            An optional argument that specifies an emoji to use as
            the icon for the link. Shortcodes are not allowed. Please use a
            single character instead. E.g. "🚨", "🔥", "🤖", etc.
            Defaults to ``None``, which means no icon is displayed.
        help : str
            An optional tooltip that gets displayed when the link is
            hovered over.
        disabled : bool
            An optional boolean, which disables the page link if set to
            ``True``. The default is ``False``.
        use_container_width : bool
            An optional boolean, which makes the link stretch its width to
            match the parent container. The default is ``True`` for page links
            in the sidebar, and ``False`` for those in the main app.

        Example
        -------
        Consider the following example given this file structure:

        >>> your-repository/
        >>> ├── pages/
        >>> │   ├── page_1.py
        >>> │   └── page_2.py
        >>> └── your_app.py

        >>> import streamlit as st
        >>>
        >>> st.page_link("your_app.py", label="Home", icon="🏠")
        >>> st.page_link("pages/page_1.py", label="Page 1", icon="1️⃣")
        >>> st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)
        >>> st.page_link("http://www.google.com", label="Google", icon="🌎")

        The default navigation is shown here for comparison, but you can hide
        the default navigation using the |client.showSidebarNavigation|_
        configuration option. This allows you to create custom, dynamic
        navigation menus for your apps!

        .. |client.showSidebarNavigation| replace:: ``client.showSidebarNavigation``
        .. _client.showSidebarNavigation: https://docs.streamlit.io/library            /advanced-features/configuration#client

        .. output ::
            https://doc-page-link.streamlit.app/
            height: 350px

        """
    return self._page_link(page=page, label=label, icon=icon, help=help, disabled=disabled, use_container_width=use_container_width)