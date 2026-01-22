def show_edges(edges):
    alledges = set(edges) | {(v, u) for u, v in edges}
    return lambda u, v: (u, v) in alledges