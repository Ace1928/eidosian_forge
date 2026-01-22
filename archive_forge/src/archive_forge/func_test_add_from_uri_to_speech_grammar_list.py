import pytest
import panel as pn
from panel.widgets import Grammar, GrammarList, SpeechToText
from panel.widgets.speech_to_text import (
def test_add_from_uri_to_speech_grammar_list():
    uri = 'http://www.example.com/grammar.txt'
    weight = 0.5
    grammar_list = GrammarList()
    result = grammar_list.add_from_uri(uri, weight)
    serialized = grammar_list.serialize()
    assert isinstance(result, Grammar)
    assert result in grammar_list
    assert serialized == [{'uri': uri, 'weight': weight}]