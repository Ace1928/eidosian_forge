from ..fix import TrainingSetCreator
def test_TrainingSetCreator_inputs():
    input_map = dict(mel_icas_in=dict(argstr='%s', copyfile=False, position=-1))
    inputs = TrainingSetCreator.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value