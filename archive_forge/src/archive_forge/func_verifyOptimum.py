from collections import Counter
from itertools import combinations, repeat
import networkx as nx
from networkx.utils import not_implemented_for
def verifyOptimum():
    if maxcardinality:
        vdualoffset = max(0, -min(dualvar.values()))
    else:
        vdualoffset = 0
    assert min(dualvar.values()) + vdualoffset >= 0
    assert len(blossomdual) == 0 or min(blossomdual.values()) >= 0
    for i, j, d in G.edges(data=True):
        wt = d.get(weight, 1)
        if i == j:
            continue
        s = dualvar[i] + dualvar[j] - 2 * wt
        iblossoms = [i]
        jblossoms = [j]
        while blossomparent[iblossoms[-1]] is not None:
            iblossoms.append(blossomparent[iblossoms[-1]])
        while blossomparent[jblossoms[-1]] is not None:
            jblossoms.append(blossomparent[jblossoms[-1]])
        iblossoms.reverse()
        jblossoms.reverse()
        for bi, bj in zip(iblossoms, jblossoms):
            if bi != bj:
                break
            s += 2 * blossomdual[bi]
        assert s >= 0
        if mate.get(i) == j or mate.get(j) == i:
            assert mate[i] == j and mate[j] == i
            assert s == 0
    for v in gnodes:
        assert v in mate or dualvar[v] + vdualoffset == 0
    for b in blossomdual:
        if blossomdual[b] > 0:
            assert len(b.edges) % 2 == 1
            for i, j in b.edges[1::2]:
                assert mate[i] == j and mate[j] == i