import pytest
from spacy import util
from spacy.tokens import Doc
from spacy.vocab import Vocab
def test_create_from_words_and_text(vocab):
    words = ["'", 'dogs', "'", 'run']
    text = "  'dogs'\n\nrun  "
    words, spaces = util.get_words_and_spaces(words, text)
    doc = Doc(vocab, words=words, spaces=spaces)
    assert [t.text for t in doc] == ['  ', "'", 'dogs', "'", '\n\n', 'run', ' ']
    assert [t.whitespace_ for t in doc] == ['', '', '', '', '', ' ', '']
    assert doc.text == text
    assert [t.text for t in doc if not t.text.isspace()] == [word for word in words if not word.isspace()]
    words = ['  ', "'", 'dogs', "'", '\n\n', 'run', ' ']
    text = "  'dogs'\n\nrun  "
    words, spaces = util.get_words_and_spaces(words, text)
    doc = Doc(vocab, words=words, spaces=spaces)
    assert [t.text for t in doc] == ['  ', "'", 'dogs', "'", '\n\n', 'run', ' ']
    assert [t.whitespace_ for t in doc] == ['', '', '', '', '', ' ', '']
    assert doc.text == text
    assert [t.text for t in doc if not t.text.isspace()] == [word for word in words if not word.isspace()]
    words = [' ', ' ', "'", 'dogs', "'", '\n\n', 'run']
    text = "  'dogs'\n\nrun  "
    words, spaces = util.get_words_and_spaces(words, text)
    doc = Doc(vocab, words=words, spaces=spaces)
    assert [t.text for t in doc] == ['  ', "'", 'dogs', "'", '\n\n', 'run', ' ']
    assert [t.whitespace_ for t in doc] == ['', '', '', '', '', ' ', '']
    assert doc.text == text
    assert [t.text for t in doc if not t.text.isspace()] == [word for word in words if not word.isspace()]
    with pytest.raises(ValueError):
        words = [' ', ' ', "'", 'dogs', "'", '\n\n', 'run']
        text = "  'dogs'\n\nrun  "
        words, spaces = util.get_words_and_spaces(words + ['away'], text)