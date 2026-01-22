from ..preprocess import MRTrixViewer
def test_MRTrixViewer_outputs():
    output_map = dict()
    outputs = MRTrixViewer.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value