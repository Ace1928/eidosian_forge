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
def filter_fragment_entries(self, fragment_entries: list) -> None:
    """
        Filter the fragment entries.

        Args:
            fragment_entries (List): Fragment entries to be filtered.
        """
    self.filtered_entries: list = []
    for entry in fragment_entries:
        if 'pcm_dielectric' in self.molecule_entry:
            err_msg = f'Principle molecule has a PCM dielectric of {self.molecule_entry['pcm_dielectric']} but a fragment entry has [[placeholder]] PCM dielectric! Please only pass fragment entries with PCM details consistent with the principle entry. Exiting...'
            if 'pcm_dielectric' not in entry:
                raise RuntimeError(err_msg.replace('[[placeholder]]', 'no'))
            if entry['pcm_dielectric'] != self.molecule_entry['pcm_dielectric']:
                raise RuntimeError(err_msg.replace('[[placeholder]]', 'a different'))
        entry['initial_molgraph'] = MoleculeGraph.from_local_env_strategy(Molecule.from_dict(entry['initial_molecule']), OpenBabelNN())
        entry['final_molgraph'] = MoleculeGraph.from_local_env_strategy(Molecule.from_dict(entry['final_molecule']), OpenBabelNN())
        if entry['initial_molgraph'].isomorphic_to(entry['final_molgraph']):
            entry['structure_change'] = 'no_change'
        else:
            initial_graph = entry['initial_molgraph'].graph
            final_graph = entry['final_molgraph'].graph
            if nx.is_connected(initial_graph.to_undirected()) and (not nx.is_connected(final_graph.to_undirected())):
                entry['structure_change'] = 'unconnected_fragments'
            elif final_graph.number_of_edges() < initial_graph.number_of_edges():
                entry['structure_change'] = 'fewer_bonds'
            elif final_graph.number_of_edges() > initial_graph.number_of_edges():
                entry['structure_change'] = 'more_bonds'
            else:
                entry['structure_change'] = 'bond_change'
        found_similar_entry = False
        for idx, filtered_entry in enumerate(self.filtered_entries):
            if filtered_entry['formula_pretty'] == entry['formula_pretty'] and (filtered_entry['initial_molgraph'].isomorphic_to(entry['initial_molgraph']) and filtered_entry['final_molgraph'].isomorphic_to(entry['final_molgraph']) and (filtered_entry['initial_molecule']['charge'] == entry['initial_molecule']['charge'])):
                found_similar_entry = True
                if entry['final_energy'] < filtered_entry['final_energy']:
                    self.filtered_entries[idx] = entry
                break
        if not found_similar_entry:
            self.filtered_entries += [entry]