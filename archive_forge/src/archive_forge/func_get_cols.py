from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Set, Union, Tuple
from triad.collections import IndexedOrderedDict
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.schema import unquote_name
def get_cols(self, *exprs: Any, ensure_distinct: bool=False, ensure_single_df: bool=False) -> List[Tuple[str, str]]:
    res: List[Tuple[str, str]] = []
    for e in exprs:
        res += self._get_cols(e)
    if ensure_distinct:
        assert_or_throw(len(set(res)) == len(res), f'there are duplicates {res}')
    if ensure_single_df:
        assert_or_throw(len(set((x[0] for x in res))) == 1, f'not from single dataframe {res}')
    return res