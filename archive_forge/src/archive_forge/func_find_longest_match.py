from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def find_longest_match(self, alo=0, ahi=None, blo=0, bhi=None):
    """Find longest matching block in a[alo:ahi] and b[blo:bhi].

        By default it will find the longest match in the entirety of a and b.

        If isjunk is not defined:

        Return (i,j,k) such that a[i:i+k] is equal to b[j:j+k], where
            alo <= i <= i+k <= ahi
            blo <= j <= j+k <= bhi
        and for all (i',j',k') meeting those conditions,
            k >= k'
            i <= i'
            and if i == i', j <= j'

        In other words, of all maximal matching blocks, return one that
        starts earliest in a, and of all those maximal matching blocks that
        start earliest in a, return the one that starts earliest in b.

        >>> s = SequenceMatcher(None, " abcd", "abcd abcd")
        >>> s.find_longest_match(0, 5, 0, 9)
        Match(a=0, b=4, size=5)

        If isjunk is defined, first the longest matching block is
        determined as above, but with the additional restriction that no
        junk element appears in the block.  Then that block is extended as
        far as possible by matching (only) junk elements on both sides.  So
        the resulting block never matches on junk except as identical junk
        happens to be adjacent to an "interesting" match.

        Here's the same example as before, but considering blanks to be
        junk.  That prevents " abcd" from matching the " abcd" at the tail
        end of the second sequence directly.  Instead only the "abcd" can
        match, and matches the leftmost "abcd" in the second sequence:

        >>> s = SequenceMatcher(lambda x: x==" ", " abcd", "abcd abcd")
        >>> s.find_longest_match(0, 5, 0, 9)
        Match(a=1, b=0, size=4)

        If no blocks match, return (alo, blo, 0).

        >>> s = SequenceMatcher(None, "ab", "c")
        >>> s.find_longest_match(0, 2, 0, 1)
        Match(a=0, b=0, size=0)
        """
    a, b, b2j, isbjunk = (self.a, self.b, self.b2j, self.bjunk.__contains__)
    if ahi is None:
        ahi = len(a)
    if bhi is None:
        bhi = len(b)
    besti, bestj, bestsize = (alo, blo, 0)
    j2len = {}
    nothing = []
    for i in range(alo, ahi):
        j2lenget = j2len.get
        newj2len = {}
        for j in b2j.get(a[i], nothing):
            if j < blo:
                continue
            if j >= bhi:
                break
            k = newj2len[j] = j2lenget(j - 1, 0) + 1
            if k > bestsize:
                besti, bestj, bestsize = (i - k + 1, j - k + 1, k)
        j2len = newj2len
    while besti > alo and bestj > blo and (not isbjunk(b[bestj - 1])) and (a[besti - 1] == b[bestj - 1]):
        besti, bestj, bestsize = (besti - 1, bestj - 1, bestsize + 1)
    while besti + bestsize < ahi and bestj + bestsize < bhi and (not isbjunk(b[bestj + bestsize])) and (a[besti + bestsize] == b[bestj + bestsize]):
        bestsize += 1
    while besti > alo and bestj > blo and isbjunk(b[bestj - 1]) and (a[besti - 1] == b[bestj - 1]):
        besti, bestj, bestsize = (besti - 1, bestj - 1, bestsize + 1)
    while besti + bestsize < ahi and bestj + bestsize < bhi and isbjunk(b[bestj + bestsize]) and (a[besti + bestsize] == b[bestj + bestsize]):
        bestsize = bestsize + 1
    return Match(besti, bestj, bestsize)