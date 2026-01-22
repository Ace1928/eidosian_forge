from ..registration import MeasureImageSimilarity
def test_MeasureImageSimilarity_outputs():
    output_map = dict(similarity=dict())
    outputs = MeasureImageSimilarity.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value