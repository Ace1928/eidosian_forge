from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def level_orbit(self, verbose=False):
    """
        Generator for all presentations obtained from this one by
        length preserving Whitehead moves. Does a depth first search
        of the orbit.  If the verbose flag is True, yields a tuple:
        (parent, Whitehead move, result, canonical presentation).
        """
    S = self.signature()
    queue = deque([(None, None, self, S)])
    seen = set([S])
    while queue:
        parent, move, pres, sig = queue.popleft()
        for a, A in pres.level_transformations():
            P = Presentation(pres.relators, pres.generators)
            P = P.whitehead_move(a, A)
            signature = P.signature()
            if signature not in seen:
                WM = (WhiteheadMove(a, A, pres.generators, self.alphabet),)
                queue.append((pres, WM, P, signature))
                seen.add(signature)
        if verbose:
            yield (parent, move, pres, Presentation(*sig))
        else:
            yield Presentation(*sig)