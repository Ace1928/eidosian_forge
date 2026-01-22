import re
import unicodedata
from nltk.tag.api import TaggerI
def set_model_file(self, model_file):
    self._model_file = model_file
    self._tagger.open(self._model_file)