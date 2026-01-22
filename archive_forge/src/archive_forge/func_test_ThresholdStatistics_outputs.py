from ..model import ThresholdStatistics
def test_ThresholdStatistics_outputs():
    output_map = dict(clusterwise_P_FDR=dict(), clusterwise_P_RF=dict(), voxelwise_P_Bonf=dict(), voxelwise_P_FDR=dict(), voxelwise_P_RF=dict(), voxelwise_P_uncor=dict())
    outputs = ThresholdStatistics.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value