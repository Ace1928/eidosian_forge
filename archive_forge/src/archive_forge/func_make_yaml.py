from __future__ import annotations
import os
import warnings
import numpy as np
import plotly.graph_objects as go
from monty.serialization import loadfn
from ruamel import yaml
from scipy.optimize import curve_fit
from pymatgen.analysis.reaction_calculator import ComputedReaction
from pymatgen.analysis.structure_analyzer import sulfide_type
from pymatgen.core import Composition, Element
def make_yaml(self, name: str='MP2020', dir: str | None=None) -> None:
    """Creates the _name_Compatibility.yaml that stores corrections as well as _name_CompatibilityUncertainties.yaml
        for correction uncertainties.

        Args:
            name: str, alternate name for the created .yaml file.
                Default: "MP2020"
            dir: str, directory in which to save the file. Pass None (default) to
                save the file in the current working directory.
        """
    if len(self.corrections) == 0:
        raise RuntimeError('Please call compute_corrections or compute_from_files to calculate corrections first')
    ggau_correction_species = ['V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'W', 'Mo']
    comp_corr: dict[str, float] = {}
    o: dict[str, float] = {}
    f: dict[str, float] = {}
    comp_corr_error: dict[str, float] = {}
    o_error: dict[str, float] = {}
    f_error: dict[str, float] = {}
    for specie in [*self.species, 'ozonide']:
        if specie in ggau_correction_species:
            o[specie] = self.corrections_dict[specie][0]
            f[specie] = self.corrections_dict[specie][0]
            o_error[specie] = self.corrections_dict[specie][1]
            f_error[specie] = self.corrections_dict[specie][1]
        else:
            comp_corr[specie] = self.corrections_dict[specie][0]
            comp_corr_error[specie] = self.corrections_dict[specie][1]
    outline = '        Name:\n        Corrections:\n            GGAUMixingCorrections:\n                O:\n                F:\n            CompositionCorrections:\n        Uncertainties:\n            GGAUMixingCorrections:\n                O:\n                F:\n            CompositionCorrections:\n        '
    fn = name + 'Compatibility.yaml'
    path = os.path.join(dir, fn) if dir else fn
    yml = yaml.YAML()
    yml.default_flow_style = False
    contents = yml.load(outline)
    contents['Name'] = name
    contents['Corrections']['GGAUMixingCorrections']['O'] = yaml.comments.CommentedMap(o)
    contents['Corrections']['GGAUMixingCorrections']['F'] = yaml.comments.CommentedMap(f)
    contents['Corrections']['CompositionCorrections'] = yaml.comments.CommentedMap(comp_corr)
    contents['Uncertainties']['GGAUMixingCorrections']['O'] = yaml.comments.CommentedMap(o_error)
    contents['Uncertainties']['GGAUMixingCorrections']['F'] = yaml.comments.CommentedMap(f_error)
    contents['Uncertainties']['CompositionCorrections'] = yaml.comments.CommentedMap(comp_corr_error)
    contents['Corrections'].yaml_set_start_comment('Energy corrections in eV/atom', indent=2)
    contents['Corrections']['GGAUMixingCorrections'].yaml_set_start_comment('Composition-based corrections applied to transition metal oxides\nand fluorides to make GGA and GGA+U energies compatible\nwhen compat_type = "Advanced" (default)', indent=4)
    contents['Corrections']['CompositionCorrections'].yaml_set_start_comment('Composition-based corrections applied to any compound containing\nthese species as anions', indent=4)
    contents['Uncertainties'].yaml_set_start_comment('Uncertainties corresponding to each energy correction (eV/atom)', indent=2)
    with open(path, mode='w') as file:
        yml.dump(contents, file)