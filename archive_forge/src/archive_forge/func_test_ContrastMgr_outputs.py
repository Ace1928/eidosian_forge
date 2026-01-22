from ..model import ContrastMgr
def test_ContrastMgr_outputs():
    output_map = dict(copes=dict(), fstats=dict(), neffs=dict(), tstats=dict(), varcopes=dict(), zfstats=dict(), zstats=dict())
    outputs = ContrastMgr.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value