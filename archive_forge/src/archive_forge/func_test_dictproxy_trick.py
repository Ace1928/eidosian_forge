import dill
from dill._dill import OLD310, MAPPING_PROXY_TRICK, DictProxyType
def test_dictproxy_trick():
    if not OLD310 and MAPPING_PROXY_TRICK:
        x = {'a': 1}
        all_views = (x.values(), x.items(), x.keys(), x)
        seperate_views = dill.copy(all_views)
        new_x = seperate_views[-1]
        new_x['b'] = 2
        new_x['c'] = 1
        assert len(new_x) == 3 and len(x) == 1
        assert len(seperate_views[0]) == 3 and len(all_views[0]) == 1
        assert len(seperate_views[1]) == 3 and len(all_views[1]) == 1
        assert len(seperate_views[2]) == 3 and len(all_views[2]) == 1
        assert dict(all_views[1]) == x
        assert dict(seperate_views[1]) == new_x