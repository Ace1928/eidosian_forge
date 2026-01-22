import pytest
import panel as pn
from panel.widgets import Grammar, GrammarList, SpeechToText
from panel.widgets.speech_to_text import (
def test_can_construct_speech_grammar_from_uri():
    uri = 'http://www.example.com/grammar.txt'
    weight = 0.7
    grammar = Grammar(uri=uri, weight=weight)
    serialized = grammar.serialize()
    assert serialized == {'uri': uri, 'weight': weight}