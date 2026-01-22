from ..nbs import NetworkBasedStatistic
def test_NetworkBasedStatistic_outputs():
    output_map = dict(nbs_network=dict(extensions=None), nbs_pval_network=dict(extensions=None), network_files=dict())
    outputs = NetworkBasedStatistic.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value