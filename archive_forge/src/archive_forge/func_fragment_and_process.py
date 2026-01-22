from __future__ import annotations
import logging
import warnings
import networkx as nx
from monty.json import MSONable
from pymatgen.analysis.fragmenter import open_ring
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
def fragment_and_process(self, bonds):
    """Fragment and process bonds.

        Args:
            bonds (list): bonds to process.
        """
    try:
        frags = self.mol_graph.split_molecule_subgraphs(bonds, allow_reverse=True)
        frag_success = True
    except MolGraphSplitError:
        if len(bonds) == 1:
            self.ring_bonds += bonds
            RO_frag = open_ring(self.mol_graph, bonds, 1000)
            frag_done = False
            for done_RO_frag in self.done_RO_frags:
                if RO_frag.isomorphic_to(done_RO_frag):
                    frag_done = True
            if not frag_done:
                self.done_RO_frags.append(RO_frag)
                opened_entries = self.search_fragment_entries(RO_frag)
                good_entries = []
                for frag in opened_entries[0]:
                    if frag['initial_molecule']['charge'] == self.molecule_entry['final_molecule']['charge']:
                        good_entries.append(frag)
                if len(good_entries) == 0:
                    for frag in opened_entries[1]:
                        if frag['initial_molecule']['charge'] == self.molecule_entry['final_molecule']['charge']:
                            good_entries.append(frag)
                if len(good_entries) == 0:
                    bb = BabelMolAdaptor.from_molecule_graph(RO_frag)
                    pb_mol = bb.pybel_mol
                    smiles = pb_mol.write('smi').split()[0]
                    specie = nx.get_node_attributes(self.mol_graph.graph, 'specie')
                    warnings.warn(f'Missing ring opening fragment resulting from the breakage of {specie[bonds[0][0]]} {specie[bonds[0][1]]} bond {bonds[0][0]} {bonds[0][1]} which would yield a molecule with this SMILES string: {smiles}')
                elif len(good_entries) == 1:
                    self.bond_dissociation_energies += [self.build_new_entry(good_entries, bonds)]
                else:
                    raise RuntimeError('There should only be one valid ring opening fragment! Exiting...')
        elif len(bonds) == 2:
            raise RuntimeError('Should only be trying to break two bonds if multibreak is true! Exiting...')
        else:
            raise ValueError('No reason to try and break more than two bonds at once! Exiting...')
        frag_success = False
    if frag_success:
        frags_done = False
        for frag_pair in self.done_frag_pairs:
            if frag_pair[0].isomorphic_to(frags[0]):
                if frag_pair[1].isomorphic_to(frags[1]):
                    frags_done = True
                    break
            elif frag_pair[1].isomorphic_to(frags[0]) and frag_pair[0].isomorphic_to(frags[1]):
                frags_done = True
                break
        if not frags_done:
            self.done_frag_pairs += [frags]
            n_entries_for_this_frag_pair = 0
            frag1_entries = self.search_fragment_entries(frags[0])
            frag2_entries = self.search_fragment_entries(frags[1])
            frag1_charges_found = []
            frag2_charges_found = []
            for frag1 in frag1_entries[0] + frag1_entries[1]:
                if frag1['initial_molecule']['charge'] not in frag1_charges_found:
                    frag1_charges_found += [frag1['initial_molecule']['charge']]
            for frag2 in frag2_entries[0] + frag2_entries[1]:
                if frag2['initial_molecule']['charge'] not in frag2_charges_found:
                    frag2_charges_found += [frag2['initial_molecule']['charge']]
            if len(frag1_charges_found) < len(self.expected_charges):
                bb = BabelMolAdaptor(frags[0].molecule)
                pb_mol = bb.pybel_mol
                smiles = pb_mol.write('smi').split()[0]
                for charge in self.expected_charges:
                    if charge not in frag1_charges_found:
                        warnings.warn(f'Missing charge={charge!r} for fragment {smiles}')
            if len(frag2_charges_found) < len(self.expected_charges):
                bb = BabelMolAdaptor(frags[1].molecule)
                pb_mol = bb.pybel_mol
                smiles = pb_mol.write('smi').split()[0]
                for charge in self.expected_charges:
                    if charge not in frag2_charges_found:
                        warnings.warn(f'Missing charge={charge!r} for fragment {smiles}')
            for frag1 in frag1_entries[0]:
                for frag2 in frag2_entries[0]:
                    if frag1['initial_molecule']['charge'] + frag2['initial_molecule']['charge'] == self.molecule_entry['final_molecule']['charge']:
                        self.bond_dissociation_energies += [self.build_new_entry([frag1, frag2], bonds)]
                        n_entries_for_this_frag_pair += 1
            if n_entries_for_this_frag_pair < len(self.expected_charges):
                for frag1 in frag1_entries[0]:
                    for frag2 in frag2_entries[1]:
                        if frag1['initial_molecule']['charge'] + frag2['initial_molecule']['charge'] == self.molecule_entry['final_molecule']['charge']:
                            self.bond_dissociation_energies += [self.build_new_entry([frag1, frag2], bonds)]
                            n_entries_for_this_frag_pair += 1
                for frag1 in frag1_entries[1]:
                    for frag2 in frag2_entries[0]:
                        if frag1['initial_molecule']['charge'] + frag2['initial_molecule']['charge'] == self.molecule_entry['final_molecule']['charge']:
                            self.bond_dissociation_energies += [self.build_new_entry([frag1, frag2], bonds)]
                            n_entries_for_this_frag_pair += 1