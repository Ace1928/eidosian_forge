from ..io import XNATSource
def test_XNATSource_outputs():
    output_map = dict()
    outputs = XNATSource.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value