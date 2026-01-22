import collections
import pickle
import pytest
import networkx as nx
from networkx.utils.configs import Config
@pytest.mark.parametrize('cfg', [EmptyConfig(), Config()])
def test_config_empty(cfg):
    assert dir(cfg) == []
    with pytest.raises(AttributeError):
        cfg.x = 1
    with pytest.raises(KeyError):
        cfg['x'] = 1
    with pytest.raises(AttributeError):
        cfg.x
    with pytest.raises(KeyError):
        cfg['x']
    assert len(cfg) == 0
    assert 'x' not in cfg
    assert cfg == cfg
    assert cfg.get('x', 2) == 2
    assert set(cfg.keys()) == set()
    assert set(cfg.values()) == set()
    assert set(cfg.items()) == set()
    cfg2 = pickle.loads(pickle.dumps(cfg))
    assert cfg == cfg2
    assert isinstance(cfg, collections.abc.Collection)
    assert isinstance(cfg, collections.abc.Mapping)