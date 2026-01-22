import networkx as nx
from collections import deque
def boundary_cycles(self):
    left = [(e[0], e, 'L') for e in self.edges]
    right = [(e[0], e, 'R') for e in self.edges]
    sides = left + right
    cycles = []
    while sides:
        cycle = []
        v, e, s = start = sides.pop()
        while True:
            cycle.append(e)
            v = e(v)
            if e.twisted and s == 'L' or (not e.twisted and (s == 'L') == (v == e[0])):
                e = self(v).succ(e)
                s = 'R' if e.twisted or v == e[0] else 'L'
            else:
                e = self(v).pred(e)
                s = 'L' if e.twisted or v == e[0] else 'R'
            if (e[0], e, s) == start:
                cycles.append(cycle)
                break
            sides.remove((e[0], e, s))
    return cycles