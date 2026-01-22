def get_neighbor_list(atoms):
    if 'neighborlist' in atoms.info['data']:
        return atoms.info['data']['neighborlist']
    else:
        return None