import difflib
from bisect import bisect
from typing import Any, Dict, List, Optional, Sequence, Tuple
def unique_lcs_py(a: Sequence[Any], b: Sequence[Any]) -> List[Tuple[int, int]]:
    """Find the longest common subset for unique lines.

    :param a: An indexable object (such as string or list of strings)
    :param b: Another indexable object (such as string or list of strings)
    :return: A list of tuples, one for each line which is matched.
            [(line_in_a, line_in_b), ...]

    This only matches lines which are unique on both sides.
    This helps prevent common lines from over influencing match
    results.
    The longest common subset uses the Patience Sorting algorithm:
    http://en.wikipedia.org/wiki/Patience_sorting
    """
    line: Any
    index: Dict[Any, Optional[int]] = {}
    for i, line in enumerate(a):
        if line in index:
            index[line] = None
        else:
            index[line] = i
    btoa: List[Optional[int]] = [None] * len(b)
    index2: Dict[Any, int] = {}
    for pos, line in enumerate(b):
        next = index.get(line)
        if next is not None:
            if line in index2:
                btoa[index2[line]] = None
                del index[line]
            else:
                index2[line] = pos
                btoa[pos] = next
    backpointers: List[Optional[int]] = [None] * len(b)
    stacks: List[int] = []
    lasts: List[int] = []
    k: int = 0
    for bpos, apos in enumerate(btoa):
        if apos is None:
            continue
        if stacks and stacks[-1] < apos:
            k = len(stacks)
        elif stacks and stacks[k] < apos and (k == len(stacks) - 1 or stacks[k + 1] > apos):
            k += 1
        else:
            k = bisect(stacks, apos)
        if k > 0:
            backpointers[bpos] = lasts[k - 1]
        if k < len(stacks):
            stacks[k] = apos
            lasts[k] = bpos
        else:
            stacks.append(apos)
            lasts.append(bpos)
    if len(lasts) == 0:
        return []
    result = []
    m: Optional[int] = lasts[-1]
    while m is not None:
        result.append((btoa[m], m))
        m = backpointers[m]
    result.reverse()
    return result