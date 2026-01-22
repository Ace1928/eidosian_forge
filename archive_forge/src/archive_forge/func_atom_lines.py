from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.utils import writer
def atom_lines(atom):
    """Generates a segment of X3D lines representing an atom."""
    x, y, z = atom.position
    lines = [(0, '<Transform translation="%.2f %.2f %.2f">' % (x, y, z))]
    lines += [(1, '<Shape>')]
    lines += [(2, '<Appearance>')]
    color = tuple(jmol_colors[atom.number])
    color = 'diffuseColor="%.3f %.3f %.3f"' % color
    lines += [(3, '<Material %s specularColor="0.5 0.5 0.5">' % color)]
    lines += [(3, '</Material>')]
    lines += [(2, '</Appearance>')]
    lines += [(2, '<Sphere radius="%.2f">' % covalent_radii[atom.number])]
    lines += [(2, '</Sphere>')]
    lines += [(1, '</Shape>')]
    lines += [(0, '</Transform>')]
    return lines