import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
def biluo_tags_to_spans(doc: Doc, tags: Iterable[str]) -> List[Span]:
    """Encode per-token tags following the BILUO scheme into Span object, e.g.
    to overwrite the doc.ents.

    doc (Doc): The document that the BILUO tags refer to.
    tags (iterable): A sequence of BILUO tags with each tag describing one
        token. Each tag string will be of the form of either "", "O" or
        "{action}-{label}", where action is one of "B", "I", "L", "U".
    RETURNS (list): A sequence of Span objects. Each token with a missing IOB
        tag is returned as a Span with an empty label.
    """
    token_offsets = tags_to_entities(tags)
    spans = []
    for label, start_idx, end_idx in token_offsets:
        span = Span(doc, start_idx, end_idx + 1, label=label)
        spans.append(span)
    return spans