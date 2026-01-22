import json
from typing import Any, Dict, List, Tuple
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_size
from triad.utils.hash import to_uuid
from triad.utils.pyarrow import SchemaedDataPartitioner
from triad.utils.schema import safe_split_out_of_quote, unquote_name
def parse_presort_exp(presort: Any) -> IndexedOrderedDict[str, bool]:
    """Returns ordered column sorting direction where ascending order
    would return as true, and descending as false.

    :param presort: string that contains column and sorting direction or
        list of tuple that contains column and boolean sorting direction
    :type presort: Any

    :return: column and boolean sorting direction of column, order matters.
    :rtype: IndexedOrderedDict[str, bool]

    .. admonition:: Examples

        >>> parse_presort_exp("b desc, c asc")
        >>> parse_presort_exp([("b", True), ("c", False))])
        both return IndexedOrderedDict([("b", True), ("c", False))])
    """
    if isinstance(presort, IndexedOrderedDict):
        return presort
    presort_list: List[Tuple[str, bool]] = []
    res: IndexedOrderedDict[str, bool] = IndexedOrderedDict()
    if presort is None:
        return res
    elif isinstance(presort, str):
        presort = presort.strip()
        if presort == '':
            return res
        for p in safe_split_out_of_quote(presort, ','):
            pp = safe_split_out_of_quote(p.strip(), ' ', max_split=1)
            key = unquote_name(pp[0].strip())
            if len(pp) == 1:
                presort_list.append((key, True))
            elif len(pp) == 2:
                if pp[1].strip().lower() == 'asc':
                    presort_list.append((key, True))
                elif pp[1].strip().lower() == 'desc':
                    presort_list.append((key, False))
                else:
                    raise SyntaxError(f'Invalid expression {presort}')
            else:
                raise SyntaxError(f'Invalid expression {presort}')
    elif isinstance(presort, list):
        for p in presort:
            if isinstance(p, str):
                presort_list.append((p, True))
            else:
                aot(len(p) == 2, SyntaxError(f'Invalid expression {presort}'))
                aot(isinstance(p, tuple) & (isinstance(p[0], str) & isinstance(p[1], bool)), SyntaxError(f'Invalid expression {presort}'))
                presort_list.append((p[0], p[1]))
    for key, value in presort_list:
        if key in res:
            raise SyntaxError(f'Invalid expression {presort} duplicated key {key}')
        res[key] = value
    return res