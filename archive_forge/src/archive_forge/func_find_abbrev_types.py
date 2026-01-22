import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def find_abbrev_types(self):
    """
        Recalculates abbreviations given type frequencies, despite no prior
        determination of abbreviations.
        This fails to include abbreviations otherwise found as "rare".
        """
    self._params.clear_abbrevs()
    tokens = (typ for typ in self._type_fdist if typ and typ.endswith('.'))
    for abbr, score, _is_add in self._reclassify_abbrev_types(tokens):
        if score >= self.ABBREV:
            self._params.abbrev_types.add(abbr)