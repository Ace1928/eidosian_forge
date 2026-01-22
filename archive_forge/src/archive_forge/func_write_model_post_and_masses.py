from ase.parallel import paropen
from ase.calculators.lammps.unitconvert import convert
def write_model_post_and_masses(fileobj, parameters):
    if 'model_post' in parameters:
        mlines = parameters['model_post']
        for ii in range(0, len(mlines)):
            fileobj.write(mlines[ii].encode('utf-8'))
    if 'masses' in parameters:
        for mass in parameters['masses']:
            fileobj.write('mass {0} \n'.format(mass).encode('utf-8'))