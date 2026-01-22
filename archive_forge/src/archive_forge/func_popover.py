from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Sequence, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.runtime.metrics_util import gather_metrics
@gather_metrics('popover')
def popover(self, label: str, *, help: str | None=None, disabled: bool=False, use_container_width: bool=False) -> 'DeltaGenerator':
    """Insert a popover container.

        Inserts a multi-element container as a popover. It consists of a button-like
        element and a container that opens when the button is clicked.

        Opening and closing the popover will not trigger a rerun. Interacting
        with widgets inside of an open popover will rerun the app while keeping
        the popover open. Clicking outside of the popover will close it.

        To add elements to the returned container, you can use the "with"
        notation (preferred) or just call methods directly on the returned object.
        See examples below.

        .. warning::
            You may not put a popover inside another popover.

        Parameters
        ----------
        label : str
            The label of the button that opens the popover container.
            The label can optionally contain Markdown and supports the
            following elements: Bold, Italics, Strikethroughs, Inline Code,
            Emojis, and Links.

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

            Unsupported elements are unwrapped so only their children (text contents) render.
            Display unsupported elements as literal characters by
            backslash-escaping them. E.g. ``1\\. Not an ordered list``.

        help : str
            An optional tooltip that gets displayed when the popover button is
            hovered over.

        disabled : bool
            An optional boolean, which disables the popover button if set to
            True. The default is False.

        use_container_width : bool
            An optional boolean, which makes the popover button stretch its width
            to match the parent container. This only affects the button and not
            the width of the popover container.

        Examples
        --------
        You can use the `with` notation to insert any element into a popover:

        >>> import streamlit as st
        >>>
        >>> with st.popover("Open popover"):
        >>>     st.markdown("Hello World 👋")
        >>>     name = st.text_input("What's your name?")
        >>>
        >>> st.write("Your name:", name)

        .. output ::
            https://doc-popover.streamlit.app/
            height: 400px

        Or you can just call methods directly on the returned objects:

        >>> import streamlit as st
        >>>
        >>> popover = st.popover("Filter items")
        >>> red = popover.checkbox("Show red items.", True)
        >>> blue = popover.checkbox("Show blue items.", True)
        >>>
        >>> if red:
        ...     st.write(":red[This is a red item.]")
        >>> if blue:
        ...     st.write(":blue[This is a blue item.]")

        .. output ::
            https://doc-popover2.streamlit.app/
            height: 400px
        """
    if label is None:
        raise StreamlitAPIException('A label is required for a popover')
    popover_proto = BlockProto.Popover()
    popover_proto.label = label
    popover_proto.use_container_width = use_container_width
    popover_proto.disabled = disabled
    if help:
        popover_proto.help = str(help)
    block_proto = BlockProto()
    block_proto.allow_empty = True
    block_proto.popover.CopyFrom(popover_proto)
    return self.dg._block(block_proto=block_proto)