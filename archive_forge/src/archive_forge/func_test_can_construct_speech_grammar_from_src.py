import pytest
import panel as pn
from panel.widgets import Grammar, GrammarList, SpeechToText
from panel.widgets.speech_to_text import (
def test_can_construct_speech_grammar_from_src():
    src = '#JSGF V1.0; grammar colors; public <color> = aqua | azure | beige;'
    weight = 0.7
    grammar = Grammar(src=src, weight=weight)
    serialized = grammar.serialize()
    assert serialized == {'src': src, 'weight': weight}