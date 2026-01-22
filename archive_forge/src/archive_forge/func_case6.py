from collections import defaultdict
def case6():
    nodes, edges, _ = case1()
    edges['I'].append('B')
    expected = {'D': None}
    return (nodes, edges, expected)