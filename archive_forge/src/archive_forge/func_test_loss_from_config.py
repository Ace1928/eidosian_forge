import numpy
import pytest
from thinc import registry
from thinc.api import (
@pytest.mark.parametrize('name,kwargs,args', [('CategoricalCrossentropy.v1', {}, (scores0, labels0)), ('SequenceCategoricalCrossentropy.v1', {}, ([scores0], [labels0])), ('CategoricalCrossentropy.v2', {'neg_prefix': '!'}, (scores0, labels0)), ('CategoricalCrossentropy.v3', {'neg_prefix': '!'}, (scores0, labels0)), ('SequenceCategoricalCrossentropy.v2', {'neg_prefix': '!'}, ([scores0], [labels0])), ('SequenceCategoricalCrossentropy.v3', {'neg_prefix': '!'}, ([scores0], [labels0])), ('L2Distance.v1', {}, (scores0, scores0)), ('CosineDistance.v1', {'normalize': True, 'ignore_zeros': True}, (scores0, scores0))])
def test_loss_from_config(name, kwargs, args):
    """Test that losses are loaded and configured correctly from registry
    (as partials)."""
    cfg = {'test': {'@losses': name, **kwargs}}
    func = registry.resolve(cfg)['test']
    loss = func.get_grad(*args)
    if isinstance(loss, (list, tuple)):
        loss = loss[0]
    assert loss.ndim == 2
    func.get_loss(*args)
    func(*args)