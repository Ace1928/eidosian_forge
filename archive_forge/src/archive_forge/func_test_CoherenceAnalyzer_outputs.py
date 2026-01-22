from ..analysis import CoherenceAnalyzer
def test_CoherenceAnalyzer_outputs():
    output_map = dict(coherence_array=dict(), coherence_csv=dict(extensions=None), coherence_fig=dict(extensions=None), timedelay_array=dict(), timedelay_csv=dict(extensions=None), timedelay_fig=dict(extensions=None))
    outputs = CoherenceAnalyzer.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value