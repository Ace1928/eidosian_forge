from ..preprocess import Realign
def test_Realign_outputs():
    output_map = dict(mean_image=dict(extensions=None), modified_in_files=dict(), realigned_files=dict(), realignment_parameters=dict())
    outputs = Realign.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value