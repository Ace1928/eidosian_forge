from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from scipy.constants import N_A
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.reaction_calculator import BalancedReaction
from pymatgen.apps.battery.battery_abc import AbstractElectrode, AbstractVoltagePair
from pymatgen.core import Composition, Element
from pymatgen.core.units import Charge, Time
def is_super_electrode(self, conversion_electrode) -> bool:
    """Checks if a particular conversion electrode is a sub electrode of the
        current electrode. Starting from a more lithiated state may result in
        a subelectrode that is essentially on the same path. For example, a
        ConversionElectrode formed by starting from an FePO4 composition would
        be a super_electrode of a ConversionElectrode formed from an LiFePO4
        composition.
        """
    for pair1 in conversion_electrode:
        rxn1 = pair1.rxn
        all_formulas1 = {rxn1.all_comp[i].reduced_formula for i in range(len(rxn1.all_comp)) if abs(rxn1.coeffs[i]) > 1e-05}
        for pair2 in self:
            rxn2 = pair2.rxn
            all_formulas2 = {rxn2.all_comp[i].reduced_formula for i in range(len(rxn2.all_comp)) if abs(rxn2.coeffs[i]) > 1e-05}
            if all_formulas1 == all_formulas2:
                break
        else:
            return False
    return True