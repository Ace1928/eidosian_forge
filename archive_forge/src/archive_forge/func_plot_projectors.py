from __future__ import annotations
import abc
import collections
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from xml.etree import ElementTree as Et
import numpy as np
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from monty.itertools import iterator_from_slice
from monty.json import MontyDecoder, MSONable
from monty.os.path import find_exts
from tabulate import tabulate
from pymatgen.core import Element
from pymatgen.core.xcfunc import XcFunc
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
@add_fig_kwargs
def plot_projectors(self, ax: plt.Axes=None, fontsize=12, **kwargs):
    """
        Plot the PAW projectors.

        Args:
            ax: matplotlib Axes or None if a new figure should be created.

        Returns:
            plt.Figure: matplotlib figure
        """
    ax, fig = get_ax_fig(ax)
    ax.grid(visible=True)
    ax.set_xlabel('r [Bohr]')
    ax.set_ylabel('$r\\tilde p\\, [Bohr]^{-\\frac{1}{2}}$')
    for state, rfunc in self.projector_functions.items():
        ax.plot(rfunc.mesh, rfunc.mesh * rfunc.values, label='TPROJ: ' + state)
    ax.legend(loc='best', shadow=True, fontsize=fontsize)
    return fig