import pytest

from falling_sand.engine.config import VoxelConfig


def test_voxel_config_defaults() -> None:
    config = VoxelConfig()
    assert config.voxel_size_m == 0.1
    assert config.chunk_size_voxels == 10
    assert config.chunk_size_m == 1.0


def test_voxel_config_validation() -> None:
    with pytest.raises(ValueError, match="voxel_size_m must be positive"):
        VoxelConfig(voxel_size_m=0.0)
    with pytest.raises(ValueError, match="chunk_size_voxels must be positive"):
        VoxelConfig(chunk_size_voxels=0)
