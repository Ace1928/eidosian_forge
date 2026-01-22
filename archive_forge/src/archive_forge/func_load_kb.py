from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple
from thinc.api import (
from thinc.types import Floats2d
from ...errors import Errors
from ...kb import (
from ...tokens import Doc, Span
from ...util import registry
from ...vocab import Vocab
from ..extract_spans import extract_spans
@registry.misc('spacy.KBFromFile.v1')
def load_kb(kb_path: Path) -> Callable[[Vocab], KnowledgeBase]:

    def kb_from_file(vocab: Vocab):
        kb = InMemoryLookupKB(vocab, entity_vector_length=1)
        kb.from_disk(kb_path)
        return kb
    return kb_from_file