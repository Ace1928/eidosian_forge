def set_parametrization(atoms, parametrization):
    if 'data' not in atoms.info:
        atoms.info['data'] = {}
    atoms.info['data']['parametrization'] = parametrization