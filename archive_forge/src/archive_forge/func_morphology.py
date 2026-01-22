import pickle
import pytest
from spacy.morphology import Morphology
from spacy.strings import StringStore
@pytest.fixture
def morphology():
    morphology = Morphology(StringStore())
    morphology.add('Feat1=Val1|Feat2=Val2')
    morphology.add('Feat3=Val3|Feat4=Val4')
    return morphology