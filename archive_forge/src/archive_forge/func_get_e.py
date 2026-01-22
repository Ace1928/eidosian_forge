def get_e(cell):
    atoms = Atoms('Au', cell=cell, pbc=1)
    atoms.calc = EMT()
    return atoms.get_potential_energy()