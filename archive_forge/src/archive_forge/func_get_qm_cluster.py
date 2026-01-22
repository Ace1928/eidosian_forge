import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
def get_qm_cluster(self, atoms):
    if self.qm_buffer_mask is None:
        self.initialize_qm_buffer_mask(atoms)
    qm_cluster = atoms[self.qm_buffer_mask]
    del qm_cluster.constraints
    round_cell = False
    if self.qm_radius is None:
        round_cell = True
        R_qm, _ = get_distances(atoms.positions[self.qm_selection_mask], cell=atoms.cell, pbc=atoms.pbc)
        self.qm_radius = np.amax(np.amax(R_qm, axis=1), axis=0) * 0.5
    if atoms.cell.orthorhombic:
        cell_size = np.diagonal(atoms.cell)
    else:
        raise RuntimeError('NON-orthorhombic cell is not supported!')
    qm_cluster_pbc = atoms.pbc & (cell_size < 2.0 * (self.qm_radius + self.buffer_width))
    qm_cluster_cell = cell_size.copy()
    qm_cluster_cell[~qm_cluster_pbc] = 2.0 * (self.qm_radius[~qm_cluster_pbc] + self.buffer_width + self.vacuum)
    if round_cell:
        qm_cluster_cell[~qm_cluster_pbc] = np.round(qm_cluster_cell[~qm_cluster_pbc] / self.qm_cell_round_off) * self.qm_cell_round_off
    qm_cluster.set_cell(Cell(np.diag(qm_cluster_cell)))
    qm_cluster.pbc = qm_cluster_pbc
    qm_shift = 0.5 * qm_cluster.cell.diagonal() - qm_cluster.positions.mean(axis=0)
    if 'cell_origin' in qm_cluster.info:
        del qm_cluster.info['cell_origin']
    qm_cluster.positions[:, ~qm_cluster_pbc] += qm_shift[~qm_cluster_pbc]
    return qm_cluster