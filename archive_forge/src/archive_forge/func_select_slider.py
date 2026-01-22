from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, Tuple, cast
from typing_extensions import TypeGuard
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Slider_pb2 import Slider as SliderProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import (
from streamlit.type_util import (
from streamlit.util import index_
@gather_metrics('select_slider')
def select_slider(self, label: str, options: OptionSequence[T]=(), value: object=None, format_func: Callable[[Any], Any]=str, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible') -> T | tuple[T, T]:
    """
        Display a slider widget to select items from a list.

        This also allows you to render a range slider by passing a two-element
        tuple or list as the ``value``.

        The difference between ``st.select_slider`` and ``st.slider`` is that
        ``select_slider`` accepts any datatype and takes an iterable set of
        options, while ``st.slider`` only accepts numerical or date/time data and
        takes a range as input.

        Parameters
        ----------
        label : str
            A short label explaining to the user what this slider is for.
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
        options : Iterable
            Labels for the select options in an Iterable. For example, this can
            be a list, numpy.ndarray, pandas.Series, pandas.DataFrame, or
            pandas.Index. For pandas.DataFrame, the first column is used.
            Each label will be cast to str internally by default.
        value : a supported type or a tuple/list of supported types or None
            The value of the slider when it first renders. If a tuple/list
            of two values is passed here, then a range slider with those lower
            and upper bounds is rendered. For example, if set to `(1, 10)` the
            slider will have a selectable range between 1 and 10.
            Defaults to first option.
        format_func : function
            Function to modify the display of the labels from the options.
            argument. It receives the option as an argument and its output
            will be cast to str.
        key : str or int
            An optional string or integer to use as the unique key for the widget.
            If this is omitted, a key will be generated for the widget
            based on its content. Multiple widgets of the same type may
            not share the same key.
        help : str
            An optional tooltip that gets displayed next to the select slider.
        on_change : callable
            An optional callback invoked when this select_slider's value changes.
        args : tuple
            An optional tuple of args to pass to the callback.
        kwargs : dict
            An optional dict of kwargs to pass to the callback.
        disabled : bool
            An optional boolean, which disables the select slider if set to True.
            The default is False.
        label_visibility : "visible", "hidden", or "collapsed"
            The visibility of the label. If "hidden", the label doesn't show but there
            is still empty space for it above the widget (equivalent to label="").
            If "collapsed", both the label and the space are removed. Default is
            "visible".

        Returns
        -------
        any value or tuple of any value
            The current value of the slider widget. The return type will match
            the data type of the value parameter.

        Examples
        --------
        >>> import streamlit as st
        >>>
        >>> color = st.select_slider(
        ...     'Select a color of the rainbow',
        ...     options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
        >>> st.write('My favorite color is', color)

        And here's an example of a range select slider:

        >>> import streamlit as st
        >>>
        >>> start_color, end_color = st.select_slider(
        ...     'Select a range of color wavelength',
        ...     options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
        ...     value=('red', 'blue'))
        >>> st.write('You selected wavelengths between', start_color, 'and', end_color)

        .. output::
           https://doc-select-slider.streamlit.app/
           height: 450px

        """
    ctx = get_script_run_ctx()
    return self._select_slider(label=label, options=options, value=value, format_func=format_func, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, disabled=disabled, label_visibility=label_visibility, ctx=ctx)