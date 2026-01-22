import itertools
import time
from typing import Dict, List, Optional, Union
from redis.client import Pipeline
from redis.utils import deprecated_function
from ..helpers import get_protocol_version, parse_to_dict
from ._util import to_string
from .aggregation import AggregateRequest, AggregateResult, Cursor
from .document import Document
from .query import Query
from .result import Result
from .suggestion import SuggestionParser
def sugget(self, key: str, prefix: str, fuzzy: bool=False, num: int=10, with_scores: bool=False, with_payloads: bool=False) -> List[SuggestionParser]:
    """
        Get a list of suggestions from the AutoCompleter, for a given prefix.

        Parameters:

        prefix : str
            The prefix we are searching. **Must be valid ascii or utf-8**
        fuzzy : bool
            If set to true, the prefix search is done in fuzzy mode.
            **NOTE**: Running fuzzy searches on short (<3 letters) prefixes
            can be very
            slow, and even scan the entire index.
        with_scores : bool
            If set to true, we also return the (refactored) score of
            each suggestion.
            This is normally not needed, and is NOT the original score
            inserted into the index.
        with_payloads : bool
            Return suggestion payloads
        num : int
            The maximum number of results we return. Note that we might
            return less. The algorithm trims irrelevant suggestions.

        Returns:

        list:
             A list of Suggestion objects. If with_scores was False, the
             score of all suggestions is 1.

        For more information see `FT.SUGGET <https://redis.io/commands/ft.sugget>`_.
        """
    args = [SUGGET_COMMAND, key, prefix, 'MAX', num]
    if fuzzy:
        args.append(FUZZY)
    if with_scores:
        args.append(WITHSCORES)
    if with_payloads:
        args.append(WITHPAYLOADS)
    res = self.execute_command(*args)
    results = []
    if not res:
        return results
    parser = SuggestionParser(with_scores, with_payloads, res)
    return [s for s in parser]