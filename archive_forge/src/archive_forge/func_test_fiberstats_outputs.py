from ..commandlineonly import fiberstats
def test_fiberstats_outputs():
    output_map = dict()
    outputs = fiberstats.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value