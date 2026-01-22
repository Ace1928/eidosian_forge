import importlib
from collections import OrderedDict
import numpy as np
import pytest
from ... import from_pystan
from ...data.io_pystan import get_draws, get_draws_stan3  # pylint: disable=unused-import
from ..helpers import (  # pylint: disable=unused-import
@pytest.mark.skipif(pystan_version() != 2, reason='PyStan 2.x required')
def test_index_order(self, data, eight_schools_params):
    """Test 0-indexed data."""
    pystan = importorskip('pystan')
    fit = data.model.sampling(data=eight_schools_params)
    if pystan.__version__ >= '2.18':
        for holder in fit.sim['samples']:
            new_chains = OrderedDict()
            for i, (key, values) in enumerate(holder.chains.items()):
                if '[' in key:
                    name, *shape = key.replace(']', '').split('[')
                    shape = [str(int(item) - 1) for items in shape for item in items.split(',')]
                    key = f'{name}[{','.join(shape)}]'
                new_chains[key] = np.full_like(values, fill_value=float(i))
            setattr(holder, 'chains', new_chains)
        fit.sim['fnames_oi'] = list(fit.sim['samples'][0].chains.keys())
    idata = from_pystan(posterior=fit)
    assert idata is not None
    for j, fpar in enumerate(fit.sim['fnames_oi']):
        par, *shape = fpar.replace(']', '').split('[')
        if par in {'lp__', 'log_lik'}:
            continue
        assert hasattr(idata.posterior, par), (par, list(idata.posterior.data_vars))
        if shape:
            shape = [slice(None), slice(None)] + list(map(int, shape))
            assert idata.posterior[par][tuple(shape)].values.mean() == float(j)
        else:
            assert idata.posterior[par].values.mean() == float(j)