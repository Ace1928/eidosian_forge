import pytest
import panel as pn
from panel.widgets import Grammar, GrammarList, SpeechToText
from panel.widgets.speech_to_text import (
def test_can_create_result_from_list():
    results = [{'is_final': True, 'alternatives': [{'confidence': 0.9190853834152222, 'transcript': 'and why'}]}]
    actual = RecognitionResult.create_from_list(results)
    assert len(actual) == 1
    assert actual[0].is_final == results[0]['is_final']
    assert len(actual[0].alternatives) == 1
    assert actual[0].alternatives[0].confidence == 0.9190853834152222
    assert actual[0].alternatives[0].transcript == 'and why'