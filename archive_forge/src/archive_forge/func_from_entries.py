from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MontyDecoder
from scipy.constants import N_A
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.apps.battery.battery_abc import AbstractElectrode, AbstractVoltagePair
from pymatgen.core import Composition, Element
from pymatgen.core.units import Charge, Time
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
@classmethod
def from_entries(cls, entry1, entry2, working_ion_entry) -> Self:
    """
        Args:
            entry1: Entry corresponding to one of the entries in the voltage step.
            entry2: Entry corresponding to the other entry in the voltage step.
            working_ion_entry: A single ComputedEntry or PDEntry representing
                the element that carries charge across the battery, e.g. Li.
        """
    working_element = working_ion_entry.elements[0]
    entry_charge = entry1
    entry_discharge = entry2
    if entry_charge.composition.get_atomic_fraction(working_element) > entry2.composition.get_atomic_fraction(working_element):
        entry_charge, entry_discharge = (entry_discharge, entry_charge)
    comp_charge = entry_charge.composition
    comp_discharge = entry_discharge.composition
    ion_sym = working_element.symbol
    frame_charge_comp = Composition({el: comp_charge[el] for el in comp_charge if el.symbol != ion_sym})
    frame_discharge_comp = Composition({el: comp_discharge[el] for el in comp_discharge if el.symbol != ion_sym})
    if not working_ion_entry.composition.is_element:
        raise ValueError('VoltagePair: The working ion specified must be an element')
    if not comp_charge.get_atomic_fraction(working_element) > 0 and (not comp_discharge.get_atomic_fraction(working_element) > 0):
        raise ValueError('VoltagePair: The working ion must be present in one of the entries')
    if comp_charge.get_atomic_fraction(working_element) == comp_discharge.get_atomic_fraction(working_element):
        raise ValueError('VoltagePair: The working ion atomic percentage cannot be the same in both the entries')
    if frame_charge_comp.reduced_formula != frame_discharge_comp.reduced_formula:
        raise ValueError('VoltagePair: the specified entries must have the same compositional framework')
    valence_list = Element(ion_sym).oxidation_states
    working_ion_valence = abs(max(valence_list))
    framework, norm_charge = frame_charge_comp.get_reduced_composition_and_factor()
    norm_discharge = frame_discharge_comp.get_reduced_composition_and_factor()[1]
    if hasattr(entry_charge, 'structure'):
        _vol_charge = entry_charge.structure.volume / norm_charge
    else:
        _vol_charge = entry_charge.data.get('volume')
    if hasattr(entry_discharge, 'structure'):
        _vol_discharge = entry_discharge.structure.volume / norm_discharge
    else:
        _vol_discharge = entry_discharge.data.get('volume')
    comp_charge = entry_charge.composition
    comp_discharge = entry_discharge.composition
    _mass_charge = comp_charge.weight / norm_charge
    _mass_discharge = comp_discharge.weight / norm_discharge
    _num_ions_transferred = comp_discharge[working_element] / norm_discharge - comp_charge[working_element] / norm_charge
    _voltage = ((entry_charge.energy / norm_charge - entry_discharge.energy / norm_discharge) / _num_ions_transferred + working_ion_entry.energy_per_atom) / working_ion_valence
    _mAh = _num_ions_transferred * Charge(1, 'e').to('C') * Time(1, 's').to('h') * N_A * 1000 * working_ion_valence
    _frac_charge = comp_charge.get_atomic_fraction(working_element)
    _frac_discharge = comp_discharge.get_atomic_fraction(working_element)
    vpair = InsertionVoltagePair(voltage=_voltage, mAh=_mAh, mass_charge=_mass_charge, mass_discharge=_mass_discharge, vol_charge=_vol_charge, vol_discharge=_vol_discharge, frac_charge=_frac_charge, frac_discharge=_frac_discharge, working_ion_entry=working_ion_entry, entry_charge=entry_charge, entry_discharge=entry_discharge, framework_formula=framework.reduced_formula)
    vpair.decomp_e_charge = entry_charge.data.get('decomposition_energy')
    vpair.decomp_e_discharge = entry_discharge.data.get('decomposition_energy')
    vpair.muO2_charge = entry_charge.data.get('muO2')
    vpair.muO2_discharge = entry_discharge.data.get('muO2')
    return vpair