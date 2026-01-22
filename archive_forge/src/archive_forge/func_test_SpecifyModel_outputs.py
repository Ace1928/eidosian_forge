from ..modelgen import SpecifyModel
def test_SpecifyModel_outputs():
    output_map = dict(session_info=dict())
    outputs = SpecifyModel.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value