from ..misc import Gunzip
def test_Gunzip_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Gunzip.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value