def json_group(t):
    m = t.get('matrix', [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    m[1], m[2], m[3], m[4], m[6], m[7], m[8], m[9], m[11], m[12], m[13], m[14] = (m[4], m[8], m[12], m[1], m[9], m[13], m[2], m[6], m[14], m[3], m[7], m[11])
    children = [sage_handlers[c['type']](c) for c in t['children']]
    return Object3d(children=children).set_matrix(m)