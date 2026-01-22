from ..convert import Camino2Trackvis
def test_Camino2Trackvis_outputs():
    output_map = dict(trackvis=dict(extensions=None))
    outputs = Camino2Trackvis.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value