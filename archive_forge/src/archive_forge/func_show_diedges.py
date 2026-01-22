def show_diedges(edges):
    edges = {(u, v) for u, v in edges}
    return lambda u, v: (u, v) in edges