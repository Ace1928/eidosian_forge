from ..convert import MergeCNetworks
def test_MergeCNetworks_outputs():
    output_map = dict(connectome_file=dict(extensions=None))
    outputs = MergeCNetworks.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value