def json_object(t):
    if t['geometry']['type'] == 'text':
        mesh = sage_handlers['text'](t)
    elif t['geometry']['type'] == 'point':
        mesh = sage_handlers['point'](t)
    elif t['geometry']['type'] == 'line':
        mesh = sage_handlers['line'](t)
    else:
        m = sage_handlers['texture'](t['texture'])
        g = sage_handlers[t['geometry']['type']](t['geometry'])
        mesh = Mesh(geometry=g, material=m)
        if t.get('mesh', False) is True:
            wireframe_material = BasicMaterial(color=2236962, transparent=True, opacity=0.2, wireframe=True)
            mesh = Object3d(children=[mesh, Mesh(geometry=g, material=wireframe_material)])
    if t['geometry']['type'] in ('cone', 'cylinder'):
        m = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, t['geometry']['height'] / 2, 1]
        mesh = Object3d(children=[mesh]).set_matrix(m)
    return mesh