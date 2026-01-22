from collections import defaultdict
def case9():
    nodes, edges, _ = case8()
    edges['C'].append('B')
    expected = {'A': None}
    return (nodes, edges, expected)