import warnings
import re
def sequence_weirdness(text: str) -> int:
    """
    This was the name of the heuristic used in ftfy 2.x through 5.x. As an
    attempt at compatibility with external code that calls the heuristic
    directly, we redirect to our new heuristic, :func:`badness`.
    """
    warnings.warn('`sequence_weirdness()` is an old heuristic, and the current closest equivalent is `ftfy.badness.badness()`')
    return badness(text)