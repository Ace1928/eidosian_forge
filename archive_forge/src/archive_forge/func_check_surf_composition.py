def check_surf_composition(images, formula):
    for atoms in images:
        zmax = atoms.positions[:, 2].max()
        sym = atoms.symbols[abs(atoms.positions[:, 2] - zmax) < 0.01]
        red_formula, _ = sym.formula.reduce()
        assert red_formula == formula