from collections import defaultdict
def case12():
    nodes, edges, _ = case8()
    edges['C'].append('D')
    edges['D'] = ['A']
    expected = {'A': {'C'}}
    return (nodes, edges, expected)