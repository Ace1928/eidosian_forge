from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
def fix_encoding_and_explain(text: str, config: Optional[TextFixerConfig]=None, **kwargs) -> ExplainedText:
    """
    Apply the steps of ftfy that detect mojibake and fix it. Returns the fixed
    text and a list explaining what was fixed.

    This includes fixing text by encoding and decoding it in different encodings,
    as well as the subordinate fixes `restore_byte_a0`, `replace_lossy_sequences`,
    `decode_inconsistent_utf8`, and `fix_c1_controls`.

    Examples::

        >>> fix_encoding_and_explain("sÃ³")
        ExplainedText(text='só', explanation=[('encode', 'latin-1'), ('decode', 'utf-8')])

        >>> result = fix_encoding_and_explain("voilÃ le travail")
        >>> result.text
        'voilà le travail'
        >>> result.explanation
        [('encode', 'latin-1'), ('transcode', 'restore_byte_a0'), ('decode', 'utf-8')]

    """
    if config is None:
        config = TextFixerConfig()
    if isinstance(text, bytes):
        raise UnicodeError(BYTES_ERROR_TEXT)
    config = _config_from_kwargs(config, kwargs)
    if not config.fix_encoding:
        return ExplainedText(text, [])
    plan_so_far: List[ExplanationStep] = []
    while True:
        prevtext = text
        text, plan = _fix_encoding_one_step_and_explain(text, config)
        if plan is not None:
            plan_so_far.extend(plan)
        if text == prevtext:
            return ExplainedText(text, plan_so_far)