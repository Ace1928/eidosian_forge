from ..checkerboardfilter import CheckerBoardFilter
def test_CheckerBoardFilter_outputs():
    output_map = dict(outputVolume=dict(extensions=None, position=-1))
    outputs = CheckerBoardFilter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value