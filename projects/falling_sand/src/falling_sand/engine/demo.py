"""Interactive Panda3D demo for the falling sand engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, radians, sin
from typing import Any, Tuple, TYPE_CHECKING

import numpy as np

from falling_sand.engine.config import VoxelConfig
from falling_sand.engine.lighting import LightingConfig, attach_default_lights
from falling_sand.engine.chunk import Chunk
from falling_sand.engine.materials import Material
from falling_sand.engine.renderer_instancing import InstanceConfig, InstancedVoxelRenderer
from falling_sand.engine.raycast import Plane, Ray, point_to_voxel, ray_plane_intersect
from falling_sand.engine.simulation import SimulationConfig, step_world
from falling_sand.engine.streaming import ChunkStreamer, StreamConfig
from falling_sand.engine.terrain import TerrainConfig, TerrainGenerator
from falling_sand.engine.tools import erase_sphere, place_sphere
from falling_sand.engine.ui import UiOverlay
from falling_sand.engine.world import World

if TYPE_CHECKING:
    from panda3d.core import Point3, WindowProperties  # type: ignore[import-untyped]
else:  # pragma: no cover - optional dependency
    try:
        from panda3d.core import Point3, WindowProperties  # type: ignore[import-untyped]
    except ImportError:
        Point3 = None
        WindowProperties = None


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for the interactive demo."""

    chunk_coord: Tuple[int, int, int] = (0, 0, 0)
    spawn_height: int = 9
    spawn_radius: int = 3
    spawn_count: int = 30
    step_interval: float = 0.05
    render_interval: float = 0.1
    player_step: int = 1
    min_height: int = 1
    stream_radius: int = 3
    stream_cache: int = 128
    fullscreen: bool = True
    mouse_capture: bool = True
    mouse_look: bool = True
    spawn_position: Tuple[int, int] = (5000, 5000)
    terrain_config: TerrainConfig = field(default_factory=TerrainConfig)
    instance_max: int = 200_000
    camera_distance_m: float = 6.0
    camera_yaw_deg: float = 225.0
    camera_pitch_deg: float = 25.0
    camera_min_pitch_deg: float = 10.0
    camera_max_pitch_deg: float = 80.0
    camera_min_distance_m: float = 2.0
    camera_max_distance_m: float = 20.0
    mouse_sensitivity: float = 90.0
    zoom_step_m: float = 0.5
    yaw_step_deg: float = 10.0

    def __post_init__(self) -> None:
        if self.spawn_height < 0:
            raise ValueError("spawn_height must be non-negative")
        if self.spawn_radius <= 0:
            raise ValueError("spawn_radius must be positive")
        if self.spawn_count <= 0:
            raise ValueError("spawn_count must be positive")
        if self.step_interval <= 0:
            raise ValueError("step_interval must be positive")
        if self.render_interval <= 0:
            raise ValueError("render_interval must be positive")
        if self.player_step <= 0:
            raise ValueError("player_step must be positive")
        if self.min_height <= 0:
            raise ValueError("min_height must be positive")
        if self.stream_radius < 0:
            raise ValueError("stream_radius must be non-negative")
        if self.stream_cache < 0:
            raise ValueError("stream_cache must be non-negative")
        if len(self.spawn_position) != 2:
            raise ValueError("spawn_position must be XY")
        if not (0 <= self.spawn_position[0] < self.terrain_config.size_x):
            raise ValueError("spawn_position x must be within terrain bounds")
        if not (0 <= self.spawn_position[1] < self.terrain_config.size_y):
            raise ValueError("spawn_position y must be within terrain bounds")
        if self.spawn_height >= self.terrain_config.height_layers:
            raise ValueError("spawn_height must be below terrain height_layers")
        if self.instance_max <= 0:
            raise ValueError("instance_max must be positive")
        if self.camera_distance_m <= 0:
            raise ValueError("camera_distance_m must be positive")
        if self.camera_min_distance_m <= 0:
            raise ValueError("camera_min_distance_m must be positive")
        if self.camera_max_distance_m <= self.camera_min_distance_m:
            raise ValueError("camera_max_distance_m must be greater than camera_min_distance_m")
        if self.camera_min_pitch_deg >= self.camera_max_pitch_deg:
            raise ValueError("camera_min_pitch_deg must be less than camera_max_pitch_deg")
        if self.mouse_sensitivity <= 0:
            raise ValueError("mouse_sensitivity must be positive")
        if self.zoom_step_m <= 0:
            raise ValueError("zoom_step_m must be positive")
        if self.yaw_step_deg <= 0:
            raise ValueError("yaw_step_deg must be positive")


def spawn_materials(
    world: World,
    rng: np.random.Generator,
    material: Material,
    config: DemoConfig,
    center: Tuple[int, int] | None = None,
) -> int:
    """Spawn materials within a radius at the top of the chunk."""

    count = 0
    if center is None:
        base_x = config.spawn_radius
        base_y = config.spawn_radius
        max_x = config.spawn_radius * 2
        max_y = config.spawn_radius * 2
    else:
        base_x = center[0] - config.spawn_radius
        base_y = center[1] - config.spawn_radius
        max_x = center[0] + config.spawn_radius
        max_y = center[1] + config.spawn_radius
    z = config.spawn_height

    for _ in range(config.spawn_count):
        x = int(rng.integers(base_x, max_x + 1))
        y = int(rng.integers(base_y, max_y + 1))
        world.set_voxel((x, y, z), material)
        count += 1
    return count


def ground_provider(coord: Tuple[int, int, int], size: int) -> Chunk:
    """Create a chunk with a solid ground at z=0."""

    chunk = Chunk.empty(coord, size)
    if coord[2] == 0:
        chunk.data[:, :, 0] = int(Material.SOLID)
        chunk.dirty = True
    return chunk


def orbit_offset(yaw_deg: float, pitch_deg: float, distance_m: float) -> Tuple[float, float, float]:
    """Compute an orbital camera offset for yaw/pitch in degrees."""

    yaw_rad = radians(yaw_deg)
    pitch_rad = radians(pitch_deg)
    planar = distance_m * cos(pitch_rad)
    return (
        planar * sin(yaw_rad),
        planar * cos(yaw_rad),
        distance_m * sin(pitch_rad),
    )


def camera_position(target: Tuple[float, float, float], offset: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Compute camera position from target and offset."""

    return (target[0] + offset[0], target[1] + offset[1], target[2] + offset[2])


def player_height(world: World, x: int, y: int, min_height: int) -> int:
    """Compute a surface-bounded player height."""

    return max(world.surface_height(x, y, default=0) + 1, min_height)


def clamp_position(x: int, y: int, terrain: TerrainConfig) -> Tuple[int, int]:
    """Clamp an XY position to terrain bounds."""

    clamped_x = max(0, min(terrain.size_x - 1, x))
    clamped_y = max(0, min(terrain.size_y - 1, y))
    return clamped_x, clamped_y


class DemoApp:
    """Panda3D demo app with basic controls."""

    def __init__(
        self,
        voxel_config: VoxelConfig | None = None,
        sim_config: SimulationConfig | None = None,
        demo_config: DemoConfig | None = None,
    ) -> None:
        self.voxel_config = voxel_config or VoxelConfig()
        self.sim_config = sim_config or SimulationConfig()
        self.demo_config = demo_config or DemoConfig()
        self.world = World(config=self.voxel_config)
        self.terrain = TerrainGenerator(self.demo_config.terrain_config)
        self.rng = np.random.default_rng(self.sim_config.rng_seed)
        self.renderer = InstancedVoxelRenderer(
            self.voxel_config,
            instance_config=InstanceConfig(max_instances=self.demo_config.instance_max),
        )
        self.streamer = ChunkStreamer(
            self.world,
            StreamConfig(radius=self.demo_config.stream_radius, cache_limit=self.demo_config.stream_cache),
            provider=self.terrain.chunk,
        )
        self.material = Material.GRANULAR
        self.paused = False
        self._time_since_step = 0.0
        self._time_since_render = 0.0
        self._render_dirty = True
        self.overlay = UiOverlay()
        self._player = (
            self.demo_config.spawn_position[0],
            self.demo_config.spawn_position[1],
            self.demo_config.min_height,
        )
        self._move_vector = (0, 0)
        self._player_np = self.renderer.base.render.attach_new_node("player")
        self._camera_pivot = self._player_np.attach_new_node("camera-pivot")
        self._camera_yaw = self.demo_config.camera_yaw_deg
        self._camera_pitch = self.demo_config.camera_pitch_deg
        self._camera_distance = self.demo_config.camera_distance_m
        self._mouse_look = self.demo_config.mouse_look
        self._last_mouse: Tuple[float, float] | None = None

        self.renderer.base.accept("1", self._set_material, [Material.GRANULAR])
        self.renderer.base.accept("2", self._set_material, [Material.LIQUID])
        self.renderer.base.accept("3", self._set_material, [Material.GAS])
        self.renderer.base.accept("4", self._set_material, [Material.SOLID])
        self.renderer.base.accept("space", self._toggle_pause)
        self.renderer.base.accept("mouse1", self._spawn_at_mouse)
        self.renderer.base.accept("mouse3", self._erase_at_mouse)
        self.renderer.base.accept("mouse_wheel_up", self._zoom, [-1.0])
        self.renderer.base.accept("mouse_wheel_down", self._zoom, [1.0])
        self.renderer.base.accept("m", self._toggle_mouse_look)
        self.renderer.base.accept("q", self._nudge_yaw, [-1.0])
        self.renderer.base.accept("e", self._nudge_yaw, [1.0])
        self.renderer.base.accept("a", self._set_move, [(-1, 0)])
        self.renderer.base.accept("d", self._set_move, [(1, 0)])
        self.renderer.base.accept("w", self._set_move, [(0, 1)])
        self.renderer.base.accept("s", self._set_move, [(0, -1)])
        self.renderer.base.accept("a-up", self._set_move, [(0, 0)])
        self.renderer.base.accept("d-up", self._set_move, [(0, 0)])
        self.renderer.base.accept("w-up", self._set_move, [(0, 0)])
        self.renderer.base.accept("s-up", self._set_move, [(0, 0)])

        self.renderer.base.disable_mouse()
        self._configure_window()
        self.renderer.base.camera.reparent_to(self._camera_pivot)
        self.renderer.base.cam.node().get_lens().set_near_far(0.05, 500.0)
        self.renderer.base.set_background_color(0.55, 0.7, 0.9, 1.0)
        attach_default_lights(self.renderer.base.render, LightingConfig())
        self.renderer.base.task_mgr.add(self._update_task, "simulation-step")
        self.streamer.update_focus_voxel(self._player)
        self._spawn()

    def _set_material(self, material: Material) -> None:
        self.material = material

    def _set_move(self, direction: Tuple[int, int]) -> None:
        self._move_vector = direction

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    def _spawn(self) -> None:
        spawn_materials(
            self.world,
            self.rng,
            self.material,
            self.demo_config,
            center=(self._player[0], self._player[1]),
        )
        self._render_dirty = True

    def _spawn_at_mouse(self) -> None:
        if not self.renderer.base.mouseWatcherNode.has_mouse():
            return
        mouse = self.renderer.base.mouseWatcherNode.get_mouse()
        lens = self.renderer.base.cam.node().get_lens()
        origin = (0.0, 0.0, 0.0)
        direction = (0.0, 0.0, -1.0)
        near: Any = self._make_point()
        far: Any = self._make_point()
        if lens.extrude(mouse, near, far):
            cam_pos = self.renderer.base.camera.get_pos(self.renderer.base.render)
            origin = (float(cam_pos.x), float(cam_pos.y), float(cam_pos.z))
            near_point = self._point_to_tuple(near)
            far_point = self._point_to_tuple(far)
            direction = (
                float(far_point[0] - near_point[0]),
                float(far_point[1] - near_point[1]),
                float(far_point[2] - near_point[2]),
            )
        plane = Plane(
            point=(0.0, 0.0, float(self.demo_config.spawn_height) * self.voxel_config.voxel_size_m),
            normal=(0.0, 0.0, 1.0),
        )
        hit = ray_plane_intersect(Ray(origin=origin, direction=direction), plane)
        if hit is None:
            return
        voxel = point_to_voxel(hit, self.voxel_config.voxel_size_m)
        place_sphere(self.world, voxel, self.material, radius=1)
        self._render_dirty = True

    def _erase_at_mouse(self) -> None:
        if not self.renderer.base.mouseWatcherNode.has_mouse():
            return
        mouse = self.renderer.base.mouseWatcherNode.get_mouse()
        lens = self.renderer.base.cam.node().get_lens()
        origin = (0.0, 0.0, 0.0)
        direction = (0.0, 0.0, -1.0)
        near: Any = self._make_point()
        far: Any = self._make_point()
        if lens.extrude(mouse, near, far):
            cam_pos = self.renderer.base.camera.get_pos(self.renderer.base.render)
            origin = (float(cam_pos.x), float(cam_pos.y), float(cam_pos.z))
            near_point = self._point_to_tuple(near)
            far_point = self._point_to_tuple(far)
            direction = (
                float(far_point[0] - near_point[0]),
                float(far_point[1] - near_point[1]),
                float(far_point[2] - near_point[2]),
            )
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        hit = ray_plane_intersect(Ray(origin=origin, direction=direction), plane)
        if hit is None:
            return
        voxel = point_to_voxel(hit, self.voxel_config.voxel_size_m)
        erase_sphere(self.world, voxel, radius=1)
        self._render_dirty = True

    def _toggle_mouse_look(self) -> None:
        self._mouse_look = not self._mouse_look
        self._last_mouse = None

    def _zoom(self, direction: float) -> None:
        self._camera_distance = self._clamp(
            self._camera_distance + direction * self.demo_config.zoom_step_m,
            self.demo_config.camera_min_distance_m,
            self.demo_config.camera_max_distance_m,
        )

    def _nudge_yaw(self, direction: float) -> None:
        self._camera_yaw += direction * self.demo_config.yaw_step_deg

    def _update_player(self) -> None:
        step = self.demo_config.player_step
        previous = self._player
        if self._move_vector != (0, 0):
            new_x = self._player[0] + self._move_vector[0] * step
            new_y = self._player[1] + self._move_vector[1] * step
        else:
            new_x, new_y = self._player[0], self._player[1]
        new_x, new_y = clamp_position(new_x, new_y, self.demo_config.terrain_config)
        new_z = player_height(self.world, new_x, new_y, self.demo_config.min_height)
        self._player = (new_x, new_y, new_z)
        if self._player != previous:
            self.streamer.update_focus_voxel(self._player)
        self._player_np.set_pos(
            self._player[0] * self.voxel_config.voxel_size_m,
            self._player[1] * self.voxel_config.voxel_size_m,
            self._player[2] * self.voxel_config.voxel_size_m,
        )

    def _update_task(self, task: Any) -> Any:
        dt = task.dt
        self._time_since_step += dt
        self._time_since_render += dt

        self._update_player()
        self._update_camera()

        if not self.paused and self._time_since_step >= self.demo_config.step_interval:
            stats = step_world(self.world.chunks, config=self.sim_config, rng=self.rng)
            if stats.moved_voxels > 0:
                self._render_dirty = True
            self._time_since_step = 0.0

        if self._render_dirty and self._time_since_render >= self.demo_config.render_interval:
            self.renderer.render_chunks(self.world.iter_chunks())
            self._time_since_render = 0.0
            self._render_dirty = False
            self._update_overlay(task)

        return task.cont

    def _update_overlay(self, task: Any) -> None:
        fps = 0.0
        if task.dt > 0:
            fps = 1.0 / task.dt
        chunk_count = len(self.world.chunks)
        overlay_text = (
            f"fps: {fps:.1f} | chunks: {chunk_count} | material: {self.material.name} | "
            f"player: {self._player} | WASD move | 1-4 material | mouse1 place | mouse3 erase | "
            "mouse look | scroll zoom | m toggle-look"
        )
        self.overlay.set_text(overlay_text)

    @staticmethod
    def _make_point() -> Any:
        if Point3 is None:
            return (0.0, 0.0, 0.0)
        return Point3(0.0, 0.0, 0.0)

    @staticmethod
    def _point_to_tuple(point: Any) -> Tuple[float, float, float]:
        if hasattr(point, "x"):
            return (float(point.x), float(point.y), float(point.z))
        return (float(point[0]), float(point[1]), float(point[2]))

    def _update_camera(self) -> None:
        self._update_camera_angles()
        offset = orbit_offset(self._camera_yaw, self._camera_pitch, self._camera_distance)
        self.renderer.base.camera.set_pos(*offset)
        self.renderer.base.camera.look_at(self._player_np)

    def run(self) -> None:
        self.renderer.run()

    def _update_camera_angles(self) -> None:
        if not self._mouse_look or not self.renderer.base.mouseWatcherNode.has_mouse():
            return
        mouse = self.renderer.base.mouseWatcherNode.get_mouse()
        current = (float(mouse.x), float(mouse.y))
        if self._last_mouse is None:
            self._last_mouse = current
            return
        dx = current[0] - self._last_mouse[0]
        dy = current[1] - self._last_mouse[1]
        self._last_mouse = current
        self._camera_yaw += dx * self.demo_config.mouse_sensitivity
        self._camera_pitch = self._clamp(
            self._camera_pitch - dy * self.demo_config.mouse_sensitivity,
            self.demo_config.camera_min_pitch_deg,
            self.demo_config.camera_max_pitch_deg,
        )

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    def _configure_window(self) -> None:
        if WindowProperties is None or self.renderer.base.win is None:
            return
        props = WindowProperties()
        if self.demo_config.fullscreen:
            props.set_fullscreen(True)
        if self.demo_config.mouse_capture:
            props.set_cursor_hidden(True)
            props.set_mouse_mode(WindowProperties.M_relative)
        self.renderer.base.win.request_properties(props)


def run_demo() -> None:
    """Run the interactive demo."""

    app = DemoApp()
    app.run()
