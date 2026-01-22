from __future__ import annotations
from collections import namedtuple
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core import Site, Species
from pymatgen.core.tensors import SquareTensor
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.due import Doi, due
def coupling_constant(self, specie):
    """
        Computes the coupling constant C_q as defined in:
            Wasylishen R E, Ashbrook S E, Wimperis S. NMR of quadrupolar nuclei
            in solid materials[M]. John Wiley & Sons, 2012. (Chapter 3.2).

        C_q for a specific atom type for this electric field tensor:
                C_q=e*Q*V_zz/h
            h: Planck's constant
            Q: nuclear electric quadrupole moment in mb (millibarn
            e: elementary proton charge

        Args:
            specie: flexible input to specify the species at this site.
                    Can take a isotope or element string, Species object,
                    or Site object

        Returns:
            the coupling constant as a FloatWithUnit in MHz
        """
    planks_constant = FloatWithUnit(6.62607004e-34, 'm^2 kg s^-1')
    Vzz = FloatWithUnit(self.V_zz, 'V ang^-2')
    e = FloatWithUnit(-1.60217662e-19, 'C')
    if isinstance(specie, str):
        if len(specie.split('-')) > 1:
            isotope = str(specie)
            specie = Species(specie.split('-')[0])
            quad_pol_mom = specie.get_nmr_quadrupole_moment(isotope)
        else:
            specie = Species(specie)
            quad_pol_mom = specie.get_nmr_quadrupole_moment()
    elif isinstance(specie, Site):
        specie = specie.specie
        quad_pol_mom = specie.get_nmr_quadrupole_moment()
    elif isinstance(specie, Species):
        quad_pol_mom = specie.get_nmr_quadrupole_moment()
    else:
        raise ValueError('Invalid species provided for quadrupolar coupling constant calculations')
    return (e * quad_pol_mom * Vzz / planks_constant).to('MHz')