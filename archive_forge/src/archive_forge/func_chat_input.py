from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, cast
from streamlit import runtime
from streamlit.elements.form import is_in_form
from streamlit.elements.image import AtomicImage, WidthBehaviour, image_to_url
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.proto.ChatInput_pb2 import ChatInput as ChatInputProto
from streamlit.proto.Common_pb2 import StringTriggerValue as StringTriggerValueProto
from streamlit.proto.RootContainer_pb2 import RootContainer
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.string_util import is_emoji
from streamlit.type_util import Key, to_key
@gather_metrics('chat_input')
def chat_input(self, placeholder: str='Your message', *, key: Key | None=None, max_chars: int | None=None, disabled: bool=False, on_submit: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None) -> str | None:
    """Display a chat input widget.

        Parameters
        ----------
        placeholder : str
            A placeholder text shown when the chat input is empty. Defaults to
            "Your message". For accessibility reasons, you should not use an
            empty string.

        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget based on
            its content. Multiple widgets of the same type may not share the same key.

        max_chars : int or None
            The maximum number of characters that can be entered. If ``None``
            (default), there will be no maximum.

        disabled : bool
            Whether the chat input should be disabled. Defaults to ``False``.

        on_submit : callable
            An optional callback invoked when the chat input's value is submitted.

        args : tuple
            An optional tuple of args to pass to the callback.

        kwargs : dict
            An optional dict of kwargs to pass to the callback.

        Returns
        -------
        str or None
            The current (non-empty) value of the text input widget on the last
            run of the app. Otherwise, ``None``.

        Examples
        --------
        When ``st.chat_input`` is used in the main body of an app, it will be
        pinned to the bottom of the page.

        >>> import streamlit as st
        >>>
        >>> prompt = st.chat_input("Say something")
        >>> if prompt:
        ...     st.write(f"User has sent the following prompt: {prompt}")

        .. output ::
            https://doc-chat-input.streamlit.app/
            height: 350px

        The chat input can also be used inline by nesting it inside any layout
        container (container, columns, tabs, sidebar, etc). Create chat
        interfaces embedded next to other content or have multiple chat bots!

        >>> import streamlit as st
        >>>
        >>> with st.sidebar:
        >>>     messages = st.container(height=300)
        >>>     if prompt := st.chat_input("Say something"):
        >>>         messages.chat_message("user").write(prompt)
        >>>         messages.chat_message("assistant").write(f"Echo: {prompt}")

        .. output ::
            https://doc-chat-input-inline.streamlit.app/
            height: 350px
        """
    default = ''
    key = to_key(key)
    check_callback_rules(self.dg, on_submit)
    check_session_state_rules(default_value=default, key=key, writes_allowed=False)
    ctx = get_script_run_ctx()
    id = compute_widget_id('chat_input', user_key=key, key=key, placeholder=placeholder, max_chars=max_chars, page=ctx.page_script_hash if ctx else None)
    if runtime.exists():
        if is_in_form(self.dg):
            raise StreamlitAPIException("`st.chat_input()` can't be used in a `st.form()`.")
    ancestor_block_types = set(self.dg._active_dg._ancestor_block_types)
    if self.dg._active_dg._root_container == RootContainer.MAIN and (not ancestor_block_types):
        position = 'bottom'
    else:
        position = 'inline'
    chat_input_proto = ChatInputProto()
    chat_input_proto.id = id
    chat_input_proto.placeholder = str(placeholder)
    if max_chars is not None:
        chat_input_proto.max_chars = max_chars
    chat_input_proto.default = default
    serde = ChatInputSerde()
    widget_state = register_widget('chat_input', chat_input_proto, user_key=key, on_change_handler=on_submit, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    chat_input_proto.disabled = disabled
    if widget_state.value_changed and widget_state.value is not None:
        chat_input_proto.value = widget_state.value
        chat_input_proto.set_value = True
    if ctx:
        save_for_app_testing(ctx, id, widget_state.value)
    if position == 'bottom':
        from streamlit import _bottom
        _bottom._enqueue('chat_input', chat_input_proto)
    else:
        self.dg._enqueue('chat_input', chat_input_proto)
    return widget_state.value if not widget_state.value_changed else None