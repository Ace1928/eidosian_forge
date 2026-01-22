import pytest
import panel as pn
from panel.widgets import Grammar, GrammarList, SpeechToText
from panel.widgets.speech_to_text import (
@pn.depends(speech_to_text, watch=True)
def update_results_html_panel(results):
    results_as_html_panel.object = speech_to_text.results_as_html