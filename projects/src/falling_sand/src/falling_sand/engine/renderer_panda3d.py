"""Panda3D renderer for voxel data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Tuple

import numpy as np

from falling_sand.engine.chunk import Chunk
from falling_sand.engine.config import VoxelConfig
from falling_sand.engine.materials import Material, MaterialPalette

if TYPE_CHECKING:
    from panda3d.core import (  # type: ignore[import-untyped]
        Geom,
        GeomNode,
        GeomPoints,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
    )
    from direct.showbase.ShowBase import ShowBase  # type: ignore[import-untyped]
else:  # pragma: no cover - optional dependency
    try:
        from panda3d.core import (  # type: ignore[import-untyped]
            Geom,
            GeomNode,
            GeomPoints,
            GeomVertexData,
            GeomVertexFormat,
            GeomVertexWriter,
        )
        from direct.showbase.ShowBase import ShowBase  # type: ignore[import-untyped]
    except ImportError:
        Geom = GeomNode = GeomPoints = GeomVertexData = GeomVertexFormat = GeomVertexWriter = None
        ShowBase = None


@dataclass(frozen=True)
class RenderConfig:
    """Rendering configuration for Panda3D."""

    point_size: float = 4.0
    max_points: int = 200_000

    def __post_init__(self) -> None:
        if self.point_size <= 0:
            raise ValueError("point_size must be positive")
        if self.max_points <= 0:
            raise ValueError("max_points must be positive")


def chunk_points(chunk: Chunk, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return positions and material ids for non-air voxels in a chunk."""

    indices = np.argwhere(chunk.data != int(Material.AIR))
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint8)
    offset = np.array(chunk.coord, dtype=np.float32) * float(chunk.size)
    positions = (indices.astype(np.float32) + 0.5 + offset) * float(voxel_size)
    materials = chunk.data[chunk.data != int(Material.AIR)].astype(np.uint8)
    return positions, materials


def build_point_cloud(
    chunks: Iterable[Chunk],
    config: VoxelConfig,
    palette: MaterialPalette,
    render_config: RenderConfig,
) -> GeomNode:
    """Build a Panda3D point cloud from chunks."""

    if GeomVertexFormat is None:
        raise RuntimeError("Panda3D is not available. Install with 'pip install panda3d'.")

    format_ = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData("voxels", format_, Geom.UH_static)
    vwriter = GeomVertexWriter(vdata, "vertex")
    cwriter = GeomVertexWriter(vdata, "color")

    count = 0
    for chunk in chunks:
        positions, materials = chunk_points(chunk, config.voxel_size_m)
        for pos, mat_id in zip(positions, materials, strict=False):
            if count >= render_config.max_points:
                break
            color = palette.color_for(Material(int(mat_id)))
            vwriter.add_data3f(float(pos[0]), float(pos[1]), float(pos[2]))
            cwriter.add_data4f(color[0], color[1], color[2], 1.0)
            count += 1
        if count >= render_config.max_points:
            break

    prim = GeomPoints(Geom.UH_static)
    prim.add_next_vertices(count)
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("voxels")
    node.add_geom(geom)
    return node


class Panda3DRenderer:
    """Simple Panda3D renderer for voxel worlds."""

    def __init__(
        self,
        config: VoxelConfig,
        palette: MaterialPalette | None = None,
        render_config: RenderConfig | None = None,
    ) -> None:
        if ShowBase is None:
            raise RuntimeError("Panda3D is not available. Install with 'pip install panda3d'.")
        self.config = config
        self.palette = palette or MaterialPalette()
        self.render_config = render_config or RenderConfig()
        self.base = ShowBase()
        self._apply_point_size(self.render_config.point_size)
        self._node_path = None

    def _apply_point_size(self, point_size: float) -> None:
        """Apply point size across Panda3D versions."""

        if hasattr(self.base.render, "set_render_mode_thickness"):
            self.base.render.set_render_mode_thickness(point_size)
            return
        if hasattr(self.base.render, "setRenderModeThickness"):
            self.base.render.setRenderModeThickness(point_size)
            return
        if hasattr(self.base.render, "set_point_size"):
            self.base.render.set_point_size(point_size)

    def render_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Render chunks as a point cloud."""

        node = build_point_cloud(chunks, self.config, self.palette, self.render_config)
        self._node_path = self.base.render.attach_new_node(node)

    def update_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Replace the current point cloud with new chunk data."""

        if self._node_path is not None:
            self._node_path.remove_node()
            self._node_path = None
        self.render_chunks(chunks)

    def run(self) -> None:
        """Start the Panda3D application loop."""

        self.base.run()
