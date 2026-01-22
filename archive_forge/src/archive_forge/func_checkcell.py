def checkcell(cell, name):
    cell = Cell.ascell(cell)
    lat = cell.get_bravais_lattice()
    assert lat.name == name, (lat.name, name)