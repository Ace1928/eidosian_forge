"""Panda3D instanced rendering for voxel worlds."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan
from typing import TYPE_CHECKING, Iterable, Tuple

import numpy as np

from falling_sand.engine.chunk import Chunk
from falling_sand.engine.config import VoxelConfig
from falling_sand.engine.materials import Material, MaterialPalette
from eidosian_core import eidosian

if TYPE_CHECKING:
    from direct.showbase.ShowBase import ShowBase  # type: ignore[import-untyped]
    from panda3d.core import NodePath, TransparencyAttrib  # type: ignore[import-untyped]
else:  # pragma: no cover - optional dependency
    try:
        from direct.showbase.ShowBase import ShowBase  # type: ignore[import-untyped]
        from panda3d.core import TransparencyAttrib  # type: ignore[import-untyped]
    except ImportError:
        ShowBase = None
        TransparencyAttrib = None


@dataclass(frozen=True)
class InstanceConfig:
    """Configuration for instanced rendering."""

    max_instances: int = 50_000
    cube_model: str = "models/box"
    max_view_distance_m: float = 80.0
    enable_frustum_culling: bool = True

    def __post_init__(self) -> None:
        if self.max_instances <= 0:
            raise ValueError("max_instances must be positive")
        if not self.cube_model:
            raise ValueError("cube_model must be non-empty")
        if self.max_view_distance_m < 0:
            raise ValueError("max_view_distance_m must be non-negative")


@eidosian()
def iter_chunk_instances(chunk: Chunk, voxel_size: float) -> Iterable[tuple[float, float, float, int]]:
    """Yield instance positions and materials for non-air voxels in a chunk."""

    indices = np.argwhere(chunk.data != int(Material.AIR))
    if indices.size == 0:
        return
    offset = np.array(chunk.coord, dtype=np.float32) * float(chunk.size)
    for ix, iy, iz in indices:
        x = (float(ix) + 0.5 + float(offset[0])) * voxel_size
        y = (float(iy) + 0.5 + float(offset[1])) * voxel_size
        z = (float(iz) + 0.5 + float(offset[2])) * voxel_size
        yield (x, y, z, int(chunk.data[ix, iy, iz]))


@eidosian()
def iter_instances(chunks: Iterable[Chunk], voxel_size: float) -> Iterable[tuple[float, float, float, int]]:
    """Yield instance positions and materials for non-air voxels."""

    for chunk in chunks:
        yield from iter_chunk_instances(chunk, voxel_size)


@eidosian()
def chunk_center(coord: Tuple[int, int, int], size: int, voxel_size: float) -> Tuple[float, float, float]:
    """Return world-space center for a chunk."""

    base = (coord[0] * size, coord[1] * size, coord[2] * size)
    half = size * 0.5
    return (
        (base[0] + half) * voxel_size,
        (base[1] + half) * voxel_size,
        (base[2] + half) * voxel_size,
    )


@eidosian()
def chunk_bounding_radius(size: int, voxel_size: float) -> float:
    """Return bounding sphere radius for a chunk."""

    half = size * voxel_size * 0.5
    return float((half * half * 3.0) ** 0.5)


@eidosian()
def normalize(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3D vector."""

    mag = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]) ** 0.5
    if mag == 0.0:
        return (0.0, 0.0, 0.0)
    return (vec[0] / mag, vec[1] / mag, vec[2] / mag)


@eidosian()
def cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Cross product of two vectors."""

    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@eidosian()
def dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Dot product of two vectors."""

    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@eidosian()
def within_distance(
    center: Tuple[float, float, float],
    camera_pos: Tuple[float, float, float],
    max_distance: float,
) -> bool:
    """Return True if a center is within the max distance."""

    if max_distance <= 0:
        return True
    dx = center[0] - camera_pos[0]
    dy = center[1] - camera_pos[1]
    dz = center[2] - camera_pos[2]
    return dx * dx + dy * dy + dz * dz <= max_distance * max_distance


@eidosian()
def frustum_visible(
    center: Tuple[float, float, float],
    radius: float,
    camera_pos: Tuple[float, float, float],
    camera_forward: Tuple[float, float, float],
    camera_up: Tuple[float, float, float],
    fov_y_deg: float,
    aspect: float,
    near: float,
    far: float,
) -> bool:
    """Approximate frustum visibility for a bounding sphere."""

    forward = normalize(camera_forward)
    up = normalize(camera_up)
    right = normalize(cross(forward, up))
    up = normalize(cross(right, forward))

    to_center = (
        center[0] - camera_pos[0],
        center[1] - camera_pos[1],
        center[2] - camera_pos[2],
    )
    z = dot(to_center, forward)
    if z + radius < near or z - radius > far or z <= 0:
        return False
    x = dot(to_center, right)
    y = dot(to_center, up)
    fov_y = radians(fov_y_deg)
    tan_y = tan(fov_y / 2.0)
    tan_x = tan_y * aspect
    return abs(x) <= z * tan_x + radius and abs(y) <= z * tan_y + radius


class InstancedVoxelRenderer:
    """Render voxels as instanced cubes."""

    def __init__(
        self,
        config: VoxelConfig,
        palette: MaterialPalette | None = None,
        instance_config: InstanceConfig | None = None,
    ) -> None:
        if ShowBase is None:
            raise RuntimeError("Panda3D is not available. Install with 'pip install panda3d'.")
        self.config = config
        self.palette = palette or MaterialPalette()
        self.instance_config = instance_config or InstanceConfig()
        self.base = ShowBase()
        self._instances: list["NodePath"] = []
        self._chunk_nodes: dict[Tuple[int, int, int], "NodePath"] = {}
        self._model = self.base.loader.load_model(self.instance_config.cube_model)
        self._model.clear_model_nodes()
        self.base.render.set_shader_auto()

    @eidosian()
    def clear(self) -> None:
        for node in self._instances:
            node.remove_node()
        self._instances = []
        for node in self._chunk_nodes.values():
            node.remove_node()
        self._chunk_nodes = {}

    @eidosian()
    def render_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Render chunks using instanced cube nodes."""

        chunk_map = {chunk.coord: chunk for chunk in chunks}
        for coord in list(self._chunk_nodes):
            if coord not in chunk_map:
                self._chunk_nodes[coord].remove_node()
                del self._chunk_nodes[coord]

        camera_pos, camera_forward, camera_up, fov_y, aspect, near, far = self._camera_state()
        max_distance = self.instance_config.max_view_distance_m
        radius = chunk_bounding_radius(self.config.chunk_size_voxels, self.config.voxel_size_m)

        count = 0
        for coord, chunk in chunk_map.items():
            center = chunk_center(coord, self.config.chunk_size_voxels, self.config.voxel_size_m)
            visible = within_distance(center, camera_pos, max_distance)
            if (
                visible
                and self.instance_config.enable_frustum_culling
                and fov_y > 0.0
                and aspect > 0.0
            ):
                visible = frustum_visible(
                    center,
                    radius,
                    camera_pos,
                    camera_forward,
                    camera_up,
                    fov_y,
                    aspect,
                    near,
                    far,
                )
            node = self._chunk_nodes.get(coord)
            if not visible:
                if node is not None:
                    node.hide()
                continue
            if node is not None:
                node.show()
            if chunk.is_empty():
                if node is not None:
                    node.remove_node()
                    del self._chunk_nodes[coord]
                continue
            if node is None or chunk.dirty:
                if node is not None:
                    node.remove_node()
                node = self.base.render.attach_new_node(f"chunk-{coord}")
                instance_count = self._build_chunk_instances(node, chunk, count)
                if instance_count == 0:
                    node.remove_node()
                    if coord in self._chunk_nodes:
                        del self._chunk_nodes[coord]
                    continue
                self._chunk_nodes[coord] = node
                chunk.dirty = False
                count += instance_count
            else:
                count += self._count_instances(node)

    @eidosian()
    def run(self) -> None:
        """Start the Panda3D application loop."""

        self.base.run()

    def _build_chunk_instances(self, parent: "NodePath", chunk: Chunk, count: int) -> int:
        instance_count = 0
        for x, y, z, mat_id in iter_chunk_instances(chunk, self.config.voxel_size_m):
            if count + instance_count >= self.instance_config.max_instances:
                break
            instance = self._model.instance_to(parent)
            instance.set_pos(x, y, z)
            rgba = self.palette.rgba_for(Material(int(mat_id)))
            instance.set_color(rgba[0], rgba[1], rgba[2], rgba[3])
            if rgba[3] < 1.0:
                if TransparencyAttrib is not None:
                    instance.set_transparency(TransparencyAttrib.M_alpha)
                else:
                    instance.set_transparency(True)
                instance.set_depth_write(False)
                instance.set_bin("transparent", 0)
            else:
                if TransparencyAttrib is not None:
                    instance.set_transparency(TransparencyAttrib.M_none)
                else:
                    instance.set_transparency(False)
                instance.set_depth_write(True)
            instance_count += 1
        return instance_count

    def _count_instances(self, node: "NodePath") -> int:
        return node.get_num_children()

    def _camera_state(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], float, float, float, float]:
        camera_pos = (0.0, 0.0, 0.0)
        camera_forward = (0.0, 1.0, 0.0)
        camera_up = (0.0, 0.0, 1.0)
        fov_y = 0.0
        aspect = 0.0
        near = 0.1
        far = 1000.0

        if hasattr(self.base, "camera") and self.base.camera is not None:
            cam_pos = self.base.camera.get_pos(self.base.render)
            camera_pos = (float(cam_pos.x), float(cam_pos.y), float(cam_pos.z))
            cam_quat = self.base.camera.get_quat(self.base.render)
            cam_forward = cam_quat.get_forward()
            cam_up = cam_quat.get_up()
            camera_forward = (float(cam_forward.x), float(cam_forward.y), float(cam_forward.z))
            camera_up = (float(cam_up.x), float(cam_up.y), float(cam_up.z))
            lens = self.base.cam.node().get_lens()
            if lens is not None:
                fov = lens.get_fov()
                if fov is not None:
                    fov_y = float(fov[1])
                aspect = float(lens.get_aspect_ratio())
                near = float(lens.get_near())
                far = float(lens.get_far())

        return camera_pos, camera_forward, camera_up, fov_y, aspect, near, far
