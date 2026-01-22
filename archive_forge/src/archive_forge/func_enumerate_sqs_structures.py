from __future__ import annotations
import multiprocessing as multiproc
import warnings
from string import ascii_uppercase
from time import time
from typing import TYPE_CHECKING
from pymatgen.command_line.mcsqs_caller import Sqs
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
def enumerate_sqs_structures(self, cluster_space: _ClusterSpace | None=None) -> list:
    """
        Generate an SQS by enumeration of all possible arrangements.

        Adapted from icet.tools.structure_generation.generate_sqs_by_enumeration
        to accommodate multiprocessing.

        Kwargs:
            cluster_space (ClusterSpace) : ClusterSpace of the SQS search.

        Returns:
            list : a list of dicts of the form: {
                    "structure": SQS structure,
                    "objective_function": SQS objective function,
                }
        """
    cr: dict[str, tuple] = {}
    cluster_space = cluster_space or self._get_cluster_space()
    sub_lattices = cluster_space.get_sublattices(cluster_space.primitive_structure)
    for sl in sub_lattices:
        mult_factor = len(sl.indices) / len(cluster_space.primitive_structure)
        if sl.symbol in self.target_concentrations:
            sl_conc = self.target_concentrations[sl.symbol]
        else:
            sl_conc = {sl.chemical_symbols[0]: 1.0}
        for species, value in sl_conc.items():
            c = value * mult_factor
            if species in cr:
                cr[species] = (cr[species][0] + c, cr[species][1] + c)
            else:
                cr[species] = (c, c)
    c_sum = sum((c[0] for c in cr.values()))
    if abs(c_sum - 1) >= self.sqs_kwargs['tol']:
        raise ValueError(f'Site occupancies sum to {abs(c_sum - 1)} instead of 1!')
    sizes = list(range(1, self.scaling + 1)) if self.sqs_kwargs['include_smaller_cells'] else [self.scaling]
    prim = cluster_space.primitive_structure
    prim.set_pbc(self.sqs_kwargs['pbc'])
    structures = enumerate_structures(prim, sizes, cluster_space.chemical_symbols, concentration_restrictions=cr)
    chunks: list[list[Atoms]] = [[] for _ in range(self.instances)]
    proc_idx = 0
    for structure in structures:
        chunks[proc_idx].append(structure)
        proc_idx = (proc_idx + 1) % self.instances
    manager = multiproc.Manager()
    working_list = manager.list()
    processes = []
    for proc_idx in range(self.instances):
        process = multiproc.Process(target=self._get_best_sqs_from_list, args=(chunks[proc_idx], working_list))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    return list(working_list)