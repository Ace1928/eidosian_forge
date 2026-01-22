from ase.atoms import Atoms
def read_mol(fileobj):
    lines = fileobj.readlines()
    L1 = lines[3]
    if L1.rstrip().endswith('V2000'):
        natoms = int(L1[:3].strip())
    else:
        natoms = int(L1.split()[0])
    positions = []
    symbols = []
    for line in lines[4:4 + natoms]:
        x, y, z, symbol = line.split()[:4]
        symbols.append(symbol)
        positions.append([float(x), float(y), float(z)])
    return Atoms(symbols=symbols, positions=positions)