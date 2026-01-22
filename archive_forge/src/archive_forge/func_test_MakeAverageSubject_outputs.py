from ..utils import MakeAverageSubject
def test_MakeAverageSubject_outputs():
    output_map = dict(average_subject_name=dict())
    outputs = MakeAverageSubject.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value