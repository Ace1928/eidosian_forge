from ase.autoneb import AutoNEB
from ase.build import fcc211, add_adsorbate
from ase.constraints import FixAtoms
from ase.neb import NEBTools
from ase.optimize import QuasiNewton
def test_autoneb(asap3, testdir):
    EMT = asap3.EMT
    fmax = 0.02
    slab = fcc211('Pt', size=(3, 2, 2), vacuum=4.0)
    add_adsorbate(slab, 'Pt', 0.5, (-0.1, 2.7))
    slab.set_constraint(FixAtoms(range(6, 12)))
    slab.calc = EMT()
    with QuasiNewton(slab, trajectory='neb000.traj') as qn:
        qn.run(fmax=fmax)
    slab[-1].x += slab.get_cell()[0, 0]
    slab[-1].y += 2.8
    with QuasiNewton(slab, trajectory='neb001.traj') as qn:
        qn.run(fmax=fmax)
    del qn

    def attach_calculators(images):
        for i in range(len(images)):
            images[i].calc = EMT()
    autoneb = AutoNEB(attach_calculators, prefix='neb', optimizer='BFGS', n_simul=3, n_max=7, fmax=fmax, k=0.5, parallel=False, maxsteps=[50, 1000])
    autoneb.run()
    nebtools = NEBTools(autoneb.all_images)
    assert abs(nebtools.get_barrier()[0] - 0.937) < 0.001