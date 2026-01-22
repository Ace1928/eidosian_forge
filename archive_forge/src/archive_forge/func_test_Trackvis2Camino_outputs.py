from ..convert import Trackvis2Camino
def test_Trackvis2Camino_outputs():
    output_map = dict(camino=dict(extensions=None))
    outputs = Trackvis2Camino.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value