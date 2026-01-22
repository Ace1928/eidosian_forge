from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Tuple, Union, cast
from prompt_toolkit.mouse_events import MouseEvent
def to_formatted_text(value: AnyFormattedText, style: str='', auto_convert: bool=False) -> FormattedText:
    """
    Convert the given value (which can be formatted text) into a list of text
    fragments. (Which is the canonical form of formatted text.) The outcome is
    always a `FormattedText` instance, which is a list of (style, text) tuples.

    It can take a plain text string, an `HTML` or `ANSI` object, anything that
    implements `__pt_formatted_text__` or a callable that takes no arguments and
    returns one of those.

    :param style: An additional style string which is applied to all text
        fragments.
    :param auto_convert: If `True`, also accept other types, and convert them
        to a string first.
    """
    result: FormattedText | StyleAndTextTuples
    if value is None:
        result = []
    elif isinstance(value, str):
        result = [('', value)]
    elif isinstance(value, list):
        result = value
    elif hasattr(value, '__pt_formatted_text__'):
        result = cast('MagicFormattedText', value).__pt_formatted_text__()
    elif callable(value):
        return to_formatted_text(value(), style=style)
    elif auto_convert:
        result = [('', f'{value}')]
    else:
        raise ValueError(f'No formatted text. Expecting a unicode object, HTML, ANSI or a FormattedText instance. Got {value!r}')
    if style:
        result = cast(StyleAndTextTuples, [(style + ' ' + item_style, *rest) for item_style, *rest in result])
    if isinstance(result, FormattedText):
        return result
    else:
        return FormattedText(result)