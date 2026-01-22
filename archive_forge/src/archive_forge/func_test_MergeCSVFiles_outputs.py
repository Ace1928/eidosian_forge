from ..misc import MergeCSVFiles
def test_MergeCSVFiles_outputs():
    output_map = dict(csv_file=dict(extensions=None))
    outputs = MergeCSVFiles.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value