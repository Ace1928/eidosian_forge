from ..io import JSONFileGrabber
def test_JSONFileGrabber_outputs():
    output_map = dict()
    outputs = JSONFileGrabber.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value