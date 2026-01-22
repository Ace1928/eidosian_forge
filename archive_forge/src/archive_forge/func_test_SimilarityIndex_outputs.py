from ..segmentation import SimilarityIndex
def test_SimilarityIndex_outputs():
    output_map = dict()
    outputs = SimilarityIndex.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value