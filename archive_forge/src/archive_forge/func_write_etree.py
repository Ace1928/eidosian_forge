from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
def write_etree(self, celltype, cartesian=False, bandstr=False, symprec: float=0.4, angle_tolerance=5, **kwargs):
    """
        Writes the exciting input parameters to an xml object.

        Args:
            celltype (str): Choice of unit cell. Can be either the unit cell
                from self.structure ("unchanged"), the conventional cell
                ("conventional"), or the primitive unit cell ("primitive").

            cartesian (bool): Whether the atomic positions are provided in
                Cartesian or unit-cell coordinates. Default is False.

            bandstr (bool): Whether the bandstructure path along the
                HighSymmKpath is included in the input file. Only supported if the
                celltype is set to "primitive". Default is False.

            symprec (float): Tolerance for the symmetry finding. Default is 0.4.

            angle_tolerance (float): Angle tolerance for the symmetry finding.
            Default is 5.

            **kwargs: Additional parameters for the input file.

        Returns:
            ET.Element containing the input XML structure
        """
    root = ET.Element('input')
    root.set('{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation', 'http://xml.exciting-code.org/excitinginput.xsd')
    title = ET.SubElement(root, 'title')
    title.text = self.title
    if cartesian:
        structure = ET.SubElement(root, 'structure', cartesian='true', speciespath='./')
    else:
        structure = ET.SubElement(root, 'structure', speciespath='./')
    crystal = ET.SubElement(structure, 'crystal')
    ang2bohr = const.value('Angstrom star') / const.value('Bohr radius')
    crystal.set('scale', str(ang2bohr))
    finder = SpacegroupAnalyzer(self.structure, symprec=symprec, angle_tolerance=angle_tolerance)
    if celltype == 'primitive':
        new_struct = finder.get_primitive_standard_structure(international_monoclinic=False)
    elif celltype == 'conventional':
        new_struct = finder.get_conventional_standard_structure(international_monoclinic=False)
    elif celltype == 'unchanged':
        new_struct = self.structure
    else:
        raise ValueError('Type of unit cell not recognized!')
    basis = new_struct.lattice.matrix
    for idx in range(3):
        base_vec = ET.SubElement(crystal, 'basevect')
        base_vec.text = f'{basis[idx][0]:16.8f} {basis[idx][1]:16.8f} {basis[idx][2]:16.8f}'
    index = 0
    for elem in sorted(new_struct.types_of_species, key=lambda el: el.X):
        species = ET.SubElement(structure, 'species', speciesfile=elem.symbol + '.xml')
        sites = new_struct.indices_from_symbol(elem.symbol)
        for j in sites:
            fc = new_struct[j].frac_coords
            coord = f'{fc[0]:16.8f} {fc[1]:16.8f} {fc[2]:16.8f}'
            if cartesian:
                coord2 = []
                for k in range(3):
                    inter = (new_struct[j].frac_coords[k] * basis[0][k] + new_struct[j].frac_coords[k] * basis[1][k] + new_struct[j].frac_coords[k] * basis[2][k]) * ang2bohr
                    coord2.append(inter)
                coord = f'{coord2[0]:16.8f} {coord2[1]:16.8f} {coord2[2]:16.8f}'
            index = index + 1
            _ = ET.SubElement(species, 'atom', coord=coord)
    if bandstr and celltype == 'primitive':
        kpath = HighSymmKpath(new_struct, symprec=symprec, angle_tolerance=angle_tolerance)
        prop = ET.SubElement(root, 'properties')
        band_struct = ET.SubElement(prop, 'bandstructure')
        for idx in range(len(kpath.kpath['path'])):
            plot = ET.SubElement(band_struct, 'plot1d')
            path = ET.SubElement(plot, 'path', steps='100')
            for j in range(len(kpath.kpath['path'][idx])):
                symbol = kpath.kpath['path'][idx][j]
                coords = kpath.kpath['kpoints'][symbol]
                coord = f'{coords[0]:16.8f} {coords[1]:16.8f} {coords[2]:16.8f}'
                symbol_map = {'\\Gamma': 'GAMMA', '\\Sigma': 'SIGMA', '\\Delta': 'DELTA', '\\Lambda': 'LAMBDA'}
                symbol = symbol_map.get(symbol, symbol)
                _ = ET.SubElement(path, 'point', coord=coord, label=symbol)
    elif bandstr and celltype != 'primitive':
        raise ValueError('Bandstructure is only implemented for the standard primitive unit cell!')
    self._dicttoxml(kwargs, root)
    return root