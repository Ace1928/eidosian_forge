import pytest
from spacy.morphology import Morphology
from spacy.strings import StringStore, get_string_id
def test_add_morphology_with_int_ids(morphology):
    morphology.strings.add('Case')
    morphology.strings.add('gen')
    morphology.strings.add('Number')
    morphology.strings.add('sing')
    morphology.add({get_string_id('Case'): get_string_id('gen'), get_string_id('Number'): get_string_id('sing')})