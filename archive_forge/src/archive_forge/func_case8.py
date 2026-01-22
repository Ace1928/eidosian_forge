from collections import defaultdict
def case8():
    edges = {'A': ['B', 'C'], 'B': ['C'], 'C': []}
    nodes = defaultdict(list)
    nodes['A'] = ['incref']
    nodes['C'] = ['decref']
    expected = {'A': {'C'}}
    return (nodes, edges, expected)