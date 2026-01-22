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
@gather_metrics('chat_message')
def chat_message(self, name: Literal['user', 'assistant', 'ai', 'human'] | str, *, avatar: Literal['user', 'assistant'] | str | AtomicImage | None=None) -> DeltaGenerator:
    """Insert a chat message container.

        To add elements to the returned container, you can use ``with`` notation
        (preferred) or just call methods directly on the returned object. See the
        examples below.

        Parameters
        ----------
        name : "user", "assistant", "ai", "human", or str
            The name of the message author. Can be "human"/"user" or
            "ai"/"assistant" to enable preset styling and avatars.

            Currently, the name is not shown in the UI but is only set as an
            accessibility label. For accessibility reasons, you should not use
            an empty string.

        avatar : str, numpy.ndarray, or BytesIO
            The avatar shown next to the message. Can be one of:

            * A single emoji, e.g. "🧑\u200d💻", "🤖", "🦖". Shortcodes are not supported.

            * An image using one of the formats allowed for ``st.image``: path of a local
                image file; URL to fetch the image from; an SVG image; array of shape
                (w,h) or (w,h,1) for a monochrome image, (w,h,3) for a color image,
                or (w,h,4) for an RGBA image.

            If None (default), uses default icons if ``name`` is "user",
            "assistant", "ai", "human" or the first letter of the ``name`` value.

        Returns
        -------
        Container
            A single container that can hold multiple elements.

        Examples
        --------
        You can use ``with`` notation to insert any element into an expander

        >>> import streamlit as st
        >>> import numpy as np
        >>>
        >>> with st.chat_message("user"):
        ...     st.write("Hello 👋")
        ...     st.line_chart(np.random.randn(30, 3))

        .. output ::
            https://doc-chat-message-user.streamlit.app/
            height: 450px

        Or you can just call methods directly in the returned objects:

        >>> import streamlit as st
        >>> import numpy as np
        >>>
        >>> message = st.chat_message("assistant")
        >>> message.write("Hello human")
        >>> message.bar_chart(np.random.randn(30, 3))

        .. output ::
            https://doc-chat-message-user1.streamlit.app/
            height: 450px

        """
    if name is None:
        raise StreamlitAPIException('The author name is required for a chat message, please set it via the parameter `name`.')
    if avatar is None and (name.lower() in {item.value for item in PresetNames} or is_emoji(name)):
        avatar = name.lower()
    avatar_type, converted_avatar = _process_avatar_input(avatar, self.dg._get_delta_path_str())
    message_container_proto = BlockProto.ChatMessage()
    message_container_proto.name = name
    message_container_proto.avatar = converted_avatar
    message_container_proto.avatar_type = avatar_type
    block_proto = BlockProto()
    block_proto.allow_empty = True
    block_proto.chat_message.CopyFrom(message_container_proto)
    return self.dg._block(block_proto=block_proto)