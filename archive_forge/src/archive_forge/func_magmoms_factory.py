import pytest
from unittest import mock
import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool
from ase.build import bulk
@pytest.fixture(params=['random', 'ones', 'binaries'])
def magmoms_factory(rng, request):
    """Factory for generating various kinds of magnetic moments"""
    kind = request.param
    if kind == 'random':
        func = rng.rand
    elif kind == 'ones':
        func = np.ones
    elif kind == 'binaries':

        def rand_binary(x):
            return rng.randint(2, size=x)
        func = rand_binary
    else:
        raise ValueError(f'Unknown kind: {kind}')

    def _magmoms_factory(atoms):
        magmoms = func(len(atoms))
        assert len(magmoms) == len(atoms)
        return magmoms
    return _magmoms_factory