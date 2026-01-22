from collections import defaultdict
import networkx as nx
def remove_back_edges(self, e):
    u = e[0]
    while self.S and top_of_stack(self.S).lowest(self) == self.height[u]:
        P = self.S.pop()
        if P.left.low is not None:
            self.side[P.left.low] = -1
    if self.S:
        P = self.S.pop()
        while P.left.high is not None and P.left.high[1] == u:
            P.left.high = self.ref[P.left.high]
        if P.left.high is None and P.left.low is not None:
            self.ref[P.left.low] = P.right.low
            self.side[P.left.low] = -1
            P.left.low = None
        while P.right.high is not None and P.right.high[1] == u:
            P.right.high = self.ref[P.right.high]
        if P.right.high is None and P.right.low is not None:
            self.ref[P.right.low] = P.left.low
            self.side[P.right.low] = -1
            P.right.low = None
        self.S.append(P)
    if self.lowpt[e] < self.height[u]:
        hl = top_of_stack(self.S).left.high
        hr = top_of_stack(self.S).right.high
        if hl is not None and (hr is None or self.lowpt[hl] > self.lowpt[hr]):
            self.ref[e] = hl
        else:
            self.ref[e] = hr