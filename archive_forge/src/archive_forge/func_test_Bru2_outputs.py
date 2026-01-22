from ..bru2nii import Bru2
def test_Bru2_outputs():
    output_map = dict(nii_file=dict(extensions=None))
    outputs = Bru2.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value