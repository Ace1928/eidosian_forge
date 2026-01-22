def find_bridges(G):
    global cnt
    global low
    global preorder
    global bridges
    cnt = 0
    low = dict()
    pre = dict()
    bridges = []
    verts = G.vertices()
    low = dict([(v, -1) for v in verts])
    preorder = dict([(v, -1) for v in verts])
    for v in verts:
        if preorder[v] == -1:
            _recursive_bridge_finding(G, v, v)
    return bridges