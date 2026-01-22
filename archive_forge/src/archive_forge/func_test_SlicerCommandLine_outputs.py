from ..dynamic_slicer import SlicerCommandLine
def test_SlicerCommandLine_outputs():
    output_map = dict()
    outputs = SlicerCommandLine.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value