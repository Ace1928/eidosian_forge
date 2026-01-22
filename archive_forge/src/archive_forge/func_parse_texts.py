import cProfile
import itertools
import pstats
import sys
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union
import srsly
import tqdm
import typer
from wasabi import Printer, msg
from ..language import Language
from ..util import load_model
from ._util import NAME, Arg, Opt, app, debug_cli
def parse_texts(nlp: Language, texts: Sequence[str]) -> None:
    for doc in nlp.pipe(tqdm.tqdm(texts, disable=None), batch_size=16):
        pass