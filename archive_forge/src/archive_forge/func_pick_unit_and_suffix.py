from typing import Iterable, List, Optional, Tuple
def pick_unit_and_suffix(size: int, suffixes: List[str], base: int) -> Tuple[int, str]:
    """Pick a suffix and base for the given size."""
    for i, suffix in enumerate(suffixes):
        unit = base ** i
        if size < unit * base:
            break
    return (unit, suffix)