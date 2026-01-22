from __future__ import annotations
import json
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.core.spectrum import Spectrum
from pymatgen.core.structure import Structure
from pymatgen.util.plotting import add_fig_kwargs
from pymatgen.vis.plotters import SpectrumPlotter
def get_plotter(self, components: Sequence=('xx',), reim: str='reim', broad: list | float=5e-05, emin: float=0, emax: float | None=None, divs: int=500, **kwargs) -> SpectrumPlotter:
    """Return an instance of the Spectrum plotter containing the different requested components.

        Arguments:
            components: A list with the components of the dielectric tensor to plot.
                        Can be either two indexes or a string like 'xx' to plot the (0,0) component
            reim: If 're' (im) is present in the string plots the real (imaginary) part of the dielectric tensor
            broad (float): a list of broadenings or a single broadening for the phonon peaks. Defaults to 0.00005.
            emin (float): minimum energy in which to obtain the spectra. Defaults to 0.
            emax (float): maximum energy in which to obtain the spectra. Defaults to None.
            divs: number of frequency samples between emin and emax
            **kwargs: Passed to IRDielectricTensor.get_spectrum()
        """
    directions_map = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}
    reim_label = {'re': 'Re', 'im': 'Im'}
    plotter = SpectrumPlotter()
    for component in components:
        i, j = (directions_map[direction] for direction in component)
        for fstr in ('re', 'im'):
            if fstr in reim:
                label = f'{reim_label[fstr]}{{$\\epsilon_{{{'xyz'[i]}{'xyz'[j]}}}$}}'
                spectrum = self.get_spectrum(component, fstr, broad=broad, emin=emin, emax=emax, divs=divs, **kwargs)
                spectrum.XLABEL = 'Frequency (meV)'
                spectrum.YLABEL = '$\\epsilon(\\omega)$'
                plotter.add_spectrum(label, spectrum)
    return plotter