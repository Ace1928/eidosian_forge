import numpy as np
def write_py(fileobj, images):
    """Write to ASE-compatible python script."""
    fileobj.write('import numpy as np\n\n')
    fileobj.write('from ase import Atoms\n\n')
    if hasattr(images, 'get_positions'):
        images = [images]
    fileobj.write('images = [\n')
    for image in images:
        fileobj.write("    Atoms(symbols='%s',\n          pbc=np.array(%s),\n          cell=np.array(\n%s),\n          positions=np.array(\n%s)),\n" % (image.get_chemical_formula(mode='reduce'), array_to_string(image.pbc, 0), array_to_string(image.cell), array_to_string(image.positions)))
    fileobj.write(']\n')