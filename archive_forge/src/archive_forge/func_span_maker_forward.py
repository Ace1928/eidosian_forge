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
def span_maker_forward(model, docs: List[Doc], is_train) -> Tuple[Ragged, Callable]:
    ops = model.ops
    n_sents = model.attrs['n_sents']
    candidates = []
    for doc in docs:
        cands = []
        try:
            sentences = [s for s in doc.sents]
        except ValueError:
            for tok in doc:
                tok.is_sent_start = tok.i == 0
            sentences = [doc[:]]
        for ent in doc.ents:
            try:
                sent_index = sentences.index(ent.sent)
            except AttributeError:
                raise RuntimeError(Errors.E030) from None
            start_sentence = max(0, sent_index - n_sents)
            end_sentence = min(len(sentences) - 1, sent_index + n_sents)
            start_token = sentences[start_sentence].start
            end_token = sentences[end_sentence].end
            cands.append((start_token, end_token))
        candidates.append(ops.asarray2i(cands))
    lengths = model.ops.asarray1i([len(cands) for cands in candidates])
    out = Ragged(model.ops.flatten(candidates), lengths)
    return (out, lambda x: [])