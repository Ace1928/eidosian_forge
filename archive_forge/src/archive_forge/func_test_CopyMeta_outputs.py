from ..dcmstack import CopyMeta
def test_CopyMeta_outputs():
    output_map = dict(dest_file=dict(extensions=None))
    outputs = CopyMeta.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value