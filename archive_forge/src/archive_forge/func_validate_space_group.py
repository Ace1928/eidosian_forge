from ase.lattice import bravais_classes
def validate_space_group(sg):
    sg = int(sg)
    if sg < 1:
        raise ValueError('Spacegroup must be positive, but is {}'.format(sg))
    if sg > 230:
        raise ValueError('Bad spacegroup', sg)
    return sg