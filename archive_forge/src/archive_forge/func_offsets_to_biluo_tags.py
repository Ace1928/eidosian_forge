import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
def offsets_to_biluo_tags(doc: Doc, entities: Iterable[Tuple[int, int, Union[str, int]]], missing: str='O') -> List[str]:
    """Encode labelled spans into per-token tags, using the
    Begin/In/Last/Unit/Out scheme (BILUO).

    doc (Doc): The document that the entity offsets refer to. The output tags
        will refer to the token boundaries within the document.
    entities (iterable): A sequence of `(start, end, label)` triples. `start`
        and `end` should be character-offset integers denoting the slice into
        the original string.
    missing (str): The label used for missing values, e.g. if tokenization
        doesn’t align with the entity offsets. Defaults to "O".
    RETURNS (list): A list of unicode strings, describing the tags. Each tag
        string will be of the form either "", "O" or "{action}-{label}", where
        action is one of "B", "I", "L", "U". The missing label is used where the
        entity offsets don't align with the tokenization in the `Doc` object.
        The training algorithm will view these as missing values. "O" denotes a
        non-entity token. "B" denotes the beginning of a multi-token entity,
        "I" the inside of an entity of three or more tokens, and "L" the end
        of an entity of two or more tokens. "U" denotes a single-token entity.

    EXAMPLE:
        >>> text = 'I like London.'
        >>> entities = [(len('I like '), len('I like London'), 'LOC')]
        >>> doc = nlp.tokenizer(text)
        >>> tags = offsets_to_biluo_tags(doc, entities)
        >>> assert tags == ["O", "O", 'U-LOC', "O"]
    """
    tokens_in_ents: Dict[int, Tuple[int, int, Union[str, int]]] = {}
    starts = {token.idx: token.i for token in doc}
    ends = {token.idx + len(token): token.i for token in doc}
    biluo = ['-' for _ in doc]
    for start_char, end_char, label in entities:
        if not label:
            for s in starts:
                if s >= start_char and s < end_char:
                    biluo[starts[s]] = 'O'
        else:
            for token_index in range(start_char, end_char):
                if token_index in tokens_in_ents.keys():
                    raise ValueError(Errors.E103.format(span1=(tokens_in_ents[token_index][0], tokens_in_ents[token_index][1], tokens_in_ents[token_index][2]), span2=(start_char, end_char, label)))
                tokens_in_ents[token_index] = (start_char, end_char, label)
            start_token = starts.get(start_char)
            end_token = ends.get(end_char)
            if start_token is not None and end_token is not None:
                if start_token == end_token:
                    biluo[start_token] = f'U-{label}'
                else:
                    biluo[start_token] = f'B-{label}'
                    for i in range(start_token + 1, end_token):
                        biluo[i] = f'I-{label}'
                    biluo[end_token] = f'L-{label}'
    entity_chars = set()
    for start_char, end_char, label in entities:
        for i in range(start_char, end_char):
            entity_chars.add(i)
    for token in doc:
        for i in range(token.idx, token.idx + len(token)):
            if i in entity_chars:
                break
        else:
            biluo[token.i] = missing
    if '-' in biluo and missing != '-':
        ent_str = str(entities)
        warnings.warn(Warnings.W030.format(text=doc.text[:50] + '...' if len(doc.text) > 50 else doc.text, entities=ent_str[:50] + '...' if len(ent_str) > 50 else ent_str))
    return biluo