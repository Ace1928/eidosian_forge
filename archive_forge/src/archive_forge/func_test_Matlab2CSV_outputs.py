from ..misc import Matlab2CSV
def test_Matlab2CSV_outputs():
    output_map = dict(csv_files=dict())
    outputs = Matlab2CSV.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value