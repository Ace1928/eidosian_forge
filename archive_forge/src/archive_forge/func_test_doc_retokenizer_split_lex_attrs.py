import numpy
import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenizer_split_lex_attrs(en_vocab):
    """Test that retokenization also sets attributes on the lexeme if they're
    lexical attributes. For example, if a user sets IS_STOP, it should mean that
    "all tokens with that lexeme" are marked as a stop word, so the ambiguity
    here is acceptable. Also see #2390.
    """
    assert not Doc(en_vocab, words=['Los'])[0].is_stop
    assert not Doc(en_vocab, words=['Angeles'])[0].is_stop
    doc = Doc(en_vocab, words=['LosAngeles', 'start'])
    assert not doc[0].is_stop
    with doc.retokenize() as retokenizer:
        attrs = {'is_stop': [True, False]}
        heads = [(doc[0], 1), doc[1]]
        retokenizer.split(doc[0], ['Los', 'Angeles'], heads, attrs=attrs)
    assert doc[0].is_stop
    assert not doc[1].is_stop