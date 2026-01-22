from ._base import *
def multi_repeat(repeats: List[Tuple[int, str]], method: str) -> str:
    """
    Build the filter to find articles containing multiple repeated words using `repeat()`
    eg. multi_repeat([(2, "airline"), (3, "airport")], "AND") finds articles that contain the word "airline" at least
    twice and "airport" at least 3 times.
    Params
    ------
        repeats: A list of (int, str) tuples to be passed to `repeat()`. Eg. [(2, "airline"), (3, "airport")]
        method: How to combine the restrictions. Must be one of "AND" or "OR"
    """
    if method not in ['AND', 'OR']:
        raise ValueError(f'method must be one of AND or OR, not {method}')
    to_repeat = [repeat(n, keyword) for n, keyword in repeats]
    return method.join(to_repeat)