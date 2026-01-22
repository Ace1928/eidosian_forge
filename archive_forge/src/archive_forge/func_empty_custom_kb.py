from pathlib import Path
from typing import Any, Callable, Dict, Iterable
import srsly
from numpy import zeros
from thinc.api import Config
from spacy import Errors, util
from spacy.kb.kb_in_memory import InMemoryLookupKB
from spacy.util import SimpleFrozenList, ensure_path, load_model_from_config, registry
from spacy.vocab import Vocab
from ..util import make_tempdir
@registry.misc('kb_test.CustomEmptyKB.v1')
def empty_custom_kb() -> Callable[[Vocab, int], SubInMemoryLookupKB]:

    def empty_kb_factory(vocab: Vocab, entity_vector_length: int):
        return SubInMemoryLookupKB(vocab=vocab, entity_vector_length=entity_vector_length, custom_field=0)
    return empty_kb_factory