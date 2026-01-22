from ..segmentation import Atropos
def test_Atropos_outputs():
    output_map = dict(classified_image=dict(extensions=None), posteriors=dict())
    outputs = Atropos.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value