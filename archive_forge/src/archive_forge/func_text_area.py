from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, cast, overload
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.TextArea_pb2 import TextArea as TextAreaProto
from streamlit.proto.TextInput_pb2 import TextInput as TextInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import (
@gather_metrics('text_area')
def text_area(self, label: str, value: str | SupportsStr | None='', height: int | None=None, max_chars: int | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> str | None:
    """Display a multi-line text input widget.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this input is for.
            The label can optionally contain Markdown and supports the following
            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.

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

            For accessibility reasons, you should never set an empty label (label="")
            but hide it with label_visibility if needed. In the future, we may disallow
            empty labels by raising an exception.
        value : object or None
            The text value of this widget when it first renders. This will be
            cast to str internally. If ``None``, will initialize empty and
            return ``None`` until the user provides input. Defaults to empty string.
        height : int or None
            Desired height of the UI element expressed in pixels. If None, a
            default height is used.
        max_chars : int or None
            Maximum number of characters allowed in text area.
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the textarea.
        on_change : callable
            An optional callback invoked when this text_area's value changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        placeholder : str or None
            An optional string displayed when the text area is empty. If None,
            no text is displayed.
        disabled : bool
            An optional boolean, which disables the text area if set to True.
            The default is False.
        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible".

        Returns
        -------
        str or None
            The current value of the text area widget or ``None`` if no value has been
            provided by the user.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> txt = st.text_area(
        ...     "Text to analyze",
        ...     "It was the best of times, it was the worst of times, it was the age of "
        ...     "wisdom, it was the age of foolishness, it was the epoch of belief, it "
        ...     "was the epoch of incredulity, it was the season of Light, it was the "
        ...     "season of Darkness, it was the spring of hope, it was the winter of "
        ...     "despair, (...)",
        ...     )
        >>>
        >>> st.write(f'You wrote {len(txt)} characters.')

        .. output::
           https://doc-text-area.streamlit.app/
           height: 300px

        """
    ctx = get_script_run_ctx()
    return self._text_area(label=label, value=value, height=height, max_chars=max_chars, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, placeholder=placeholder, disabled=disabled, label_visibility=label_visibility, ctx=ctx)