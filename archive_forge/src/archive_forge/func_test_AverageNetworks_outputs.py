from ..nx import AverageNetworks
def test_AverageNetworks_outputs():
    output_map = dict(gexf_groupavg=dict(extensions=None), gpickled_groupavg=dict(extensions=None), matlab_groupavgs=dict())
    outputs = AverageNetworks.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value