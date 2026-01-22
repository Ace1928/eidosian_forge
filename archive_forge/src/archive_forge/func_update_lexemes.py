import logging
from pathlib import Path
from typing import Optional
import srsly
import typer
from wasabi import msg
from .. import util
from ..language import Language
from ..training.initialize import convert_vectors, init_nlp
from ._util import (
def update_lexemes(nlp: Language, jsonl_loc: Path) -> None:
    lex_attrs = srsly.read_jsonl(jsonl_loc)
    for attrs in lex_attrs:
        if 'settings' in attrs:
            continue
        lexeme = nlp.vocab[attrs['orth']]
        lexeme.set_attrs(**attrs)