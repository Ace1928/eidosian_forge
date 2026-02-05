"""Gene Particles Cellular Automata Simulation.

Provides the core simulation framework for cellular automata with emergent evolution,
interaction physics, and environmental dynamics using vectorized operations for
maximum performance with precise static typing throughout.
"""

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import pygame
from numpy.typing import NDArray
from eidosian_core import eidosian

# Local imports with explicit paths
from game_forge.src.gene_particles.gp_config import ReproductionMode, SimulationConfig
from game_forge.src.gene_particles.gp_interpreter import GeneticInterpreter
from game_forge.src.gene_particles.gp_manager import CellularTypeManager
from game_forge.src.gene_particles.gp_renderer import Renderer
from game_forge.src.gene_particles.gp_rules import InteractionRules
from game_forge.src.gene_particles.gp_types import (
    BoolArray,
    CellularTypeData,
    FloatArray,
    IntArray,
)
from game_forge.src.gene_particles.gp_utility import (
    apply_synergy,
    generate_vibrant_colors,
    give_take_interaction,
    tile_positions_for_wrap,
    wrap_deltas,
    wrap_positions,
)
from game_forge.src.gene_particles.gp_ui import SimulationUI


# Heuristic controls for interaction neighbor search
INTERACTION_KDTREE_THRESHOLD = 50000
INTERACTION_DENSE_FRACTION = 0.35


# Protocol for pygame.display.Info() return type
class PygameDisplayInfo(Protocol):
    """Type protocol for pygame.display.Info() return value."""

    current_w: int
    current_h: int


# Import scipy's KDTree with proper type handling
try:
    from scipy.spatial import cKDTree as KDTree  # type: ignore[import]
except ImportError:
    # Type stub for when scipy isn't available
    class KDTree:
        """Type stub for SciPy's KDTree spatial index."""

        def __init__(self, data: NDArray[np.float64], leafsize: int = 10) -> None:
            """Initialize KDTree with position data."""
            # Suppress unused parameter warnings with no-op
            _ = data, leafsize

        @eidosian()
        def query_ball_point(
            self, x: NDArray[np.float64], r: float, p: float = 2.0, eps: float = 0
        ) -> List[List[int]]:
            """Query for all points within distance r of x."""
            # Suppress unused parameter warnings with no-op
            _ = x, r, p, eps
            return [[]]


###############################################################
# Cellular Automata (Main Simulation)
###############################################################


class CellularAutomata:
    """
    Primary simulation controller implementing cellular automata dynamics.

    Orchestrates all simulation components including initialization, frame updates,
    inter-type interactions, clustering behaviors, reproduction, and visualization.
    Manages the full lifecycle of particles with optimized vectorized operations.

    Attributes:
        config: Master configuration parameters
        screen: Pygame display surface
        clock: Timing controller for frame rate management
        frame_count: Current simulation frame number
        run_flag: Boolean controlling main loop execution
        edge_buffer: Distance from screen edges for boundary calculations
        colors: RGB color tuples for each cellular type
        type_manager: Controller for all cellular type data
        rules_manager: Manager of interaction rules between types
        renderer: Handles visual representation of particles
        species_count: Dictionary tracking population by species ID
        screen_bounds: NumPy array of screen boundary coordinates
    """

    def __init__(
        self,
        config: SimulationConfig,
        fullscreen: bool = True,
        screen_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize the simulation environment with specified configuration.

        Sets up display, managers, and initial particle population.

        Args:
            config: Configuration parameters controlling all simulation aspects
            fullscreen: Use fullscreen display mode when True
            screen_size: Window size for non-fullscreen mode
        """
        # Core system initialization
        self.config: SimulationConfig = config
        pygame.init()

        # Display setup with optimal performance flags
        if fullscreen:
            display_info: PygameDisplayInfo = pygame.display.Info()
            screen_width = int(display_info.current_w)
            screen_height = int(display_info.current_h)
            flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
        else:
            if screen_size is None:
                screen_size = (800, 600)
            screen_width, screen_height = screen_size
            flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode((screen_width, screen_height), flags)
        pygame.display.set_caption("Emergent Cellular Automata Simulation")

        # Simulation control variables
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.frame_count: int = 0
        self.run_flag: bool = True
        self.spatial_dimensions: int = self.config.spatial_dimensions

        # Calculate screen boundaries with buffer
        self.edge_buffer: float = 0.05 * float(max(screen_width, screen_height))
        self.screen_bounds: FloatArray = np.array(
            [
                self.edge_buffer,  # Left bound
                screen_width - self.edge_buffer,  # Right bound
                self.edge_buffer,  # Top bound
                screen_height - self.edge_buffer,  # Bottom bound
            ],
            dtype=np.float64,
        )

        # World bounds for wrap mode (no buffer)
        self.world_bounds: FloatArray = np.array(
            [0.0, float(screen_width), 0.0, float(screen_height)],
            dtype=np.float64,
        )
        self.config.world_width = float(screen_width)
        self.config.world_height = float(screen_height)
        if self.spatial_dimensions == 3:
            if self.config.world_depth is None:
                self.config.world_depth = float(max(screen_width, screen_height))
            depth_max = float(self.config.world_depth)
            depth_min = 0.0
            self.depth_bounds: Optional[FloatArray] = np.array(
                [depth_min, depth_max], dtype=np.float64
            )
            self.world_size = np.array(
                [float(screen_width), float(screen_height), float(depth_max)],
                dtype=np.float64,
            )
            self.config.world_depth = depth_max
        else:
            self.depth_bounds = None
            self.world_size = np.array(
                [float(screen_width), float(screen_height)], dtype=np.float64
            )

        # Generate vibrant, visually distinct colors for particle types
        self.colors: List[Tuple[int, int, int]] = generate_vibrant_colors(
            self.config.n_cell_types
        )
        self._interaction_cache: Optional[
            Tuple[
                List[Optional[FloatArray]],
                List[Optional[KDTree]],
                List[Optional[IntArray]],
                Tuple[float, ...],
            ]
        ] = None

        # Determine which types use mass-based physics
        n_mass_types: int = int(
            self.config.mass_based_fraction * self.config.n_cell_types
        )
        mass_based_type_indices: List[int] = list(range(n_mass_types))

        # Initialize managers for particle types and interactions
        self.type_manager: CellularTypeManager = CellularTypeManager(
            self.config, self.colors, mass_based_type_indices
        )

        # Generate mass values for mass-based particle types
        mass_values: FloatArray = np.random.uniform(
            self.config.mass_range[0], self.config.mass_range[1], n_mass_types
        )

        # Create and initialize all cellular types
        for i in range(self.config.n_cell_types):
            ct = CellularTypeData(
                type_id=i,
                color=self.colors[i],
                n_particles=self.config.particles_per_type,
                window_width=screen_width,
                window_height=screen_height,
                initial_energy=self.config.initial_energy,
                max_age=self.config.max_age,
                mass=mass_values[i] if i < n_mass_types else None,
                base_velocity_scale=self.config.base_velocity_scale,
                window_depth=int(self.config.world_depth)
                if self.config.world_depth is not None
                else None,
                spatial_dimensions=self.spatial_dimensions,
            )
            self.type_manager.add_cellular_type_data(ct)

        # Initialize interaction rules and rendering system
        self.rules_manager: InteractionRules = InteractionRules(
            self.config, mass_based_type_indices
        )
        self.renderer: Renderer = Renderer(self.screen, self.config)
        self.ui: SimulationUI = SimulationUI(self.screen, self.config)

        # Genetic interpreter for gene-driven behaviors and reproduction
        self.genetic_interpreter: GeneticInterpreter = GeneticInterpreter(
            self.config.gene_sequence
        )
        self.interpreter_enabled: bool = (
            self.config.use_gene_interpreter
            or self.config.reproduction_mode != ReproductionMode.MANAGER
        )

        # Initialize species tracking with default value handling
        self.species_count: DefaultDict[int, int] = defaultdict(int)
        self.update_species_count()

    @eidosian()
    def update_species_count(self) -> None:
        """
        Update the count of unique species across all cellular types.

        Clears existing counts and recalculates population sizes for each
        species ID found across all cellular types using vectorized operations.
        Results are stored in the species_count dictionary for statistics
        and rendering.
        """
        self.species_count.clear()
        for ct in self.type_manager.cellular_types:
            unique, counts = np.unique(ct.species_id, return_counts=True)
            for species_id, count in zip(unique, counts):
                self.species_count[int(species_id)] += int(count)

    @eidosian()
    def main_loop(self) -> None:
        """
        Execute the main simulation loop until termination conditions are met.

        Each frame performs:
        1. Event handling (exit conditions, user input)
        2. Interaction rule evolution
        3. Inter-type interactions (forces, energy transfers)
        4. Clustering behaviors within types
        5. Reproduction and death management
        6. Boundary handling and rendering
        7. Performance monitoring and population control

        Loop exits on ESC key, window close, or reaching max_frames.
        """
        while self.run_flag:
            # Update frame counter and check termination conditions
            self.frame_count += 1

            # Handle Pygame events (window close, key presses)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    self.run_flag = False
                    break
                self.ui.handle_event(event, self)
            if not self.run_flag:
                break

            if self.ui.state.paused and not self.ui.state.single_step:
                stats: Dict[str, float] = {
                    "fps": self.clock.get_fps(),
                    "total_species": float(len(self.species_count)),
                    "total_particles": float(sum(self.species_count.values())),
                }
                self.screen.fill((20, 20, 30))
                for ct in self.type_manager.cellular_types:
                    self.renderer.draw_cellular_type(ct)
                self.renderer.render(stats)
                self.ui.render(stats)
                pygame.display.flip()
                self.clock.tick(60)
                continue
            if self.ui.state.single_step:
                self.ui.state.single_step = False

            # Advance environment state and run hooks
            self.config.advance_environment(self.frame_count)

            # Evolve interaction parameters periodically
            self.rules_manager.evolve_parameters(self.frame_count)

            # Clear the display with background color
            self.screen.fill((20, 20, 30))  # Dark blue-gray background

            # Apply all inter-type interactions and updates
            self.apply_all_interactions()

            # Apply genetic interpreter at configured cadence
            if (
                self.interpreter_enabled
                and self.frame_count % self.config.gene_interpreter_interval == 0
            ):
                self.apply_gene_interpreter()

            # Apply clustering within each cellular type
            for ct in self.type_manager.cellular_types:
                self.apply_clustering(ct)

            # Handle reproduction and death across all types
            if self.config.reproduction_mode in (
                ReproductionMode.MANAGER,
                ReproductionMode.HYBRID,
            ):
                self.type_manager.reproduce()
            self.type_manager.remove_dead_in_all_types()
            self.update_species_count()

            # Render all cellular types to the display
            for ct in self.type_manager.cellular_types:
                self.renderer.draw_cellular_type(ct)

            # Prepare statistics for display
            stats: Dict[str, float] = {
                "fps": self.clock.get_fps(),
                "total_species": float(len(self.species_count)),
                "total_particles": float(sum(self.species_count.values())),
            }

            # Render UI elements with current statistics
            self.renderer.render(stats)
            self.ui.render(stats)
            pygame.display.flip()

            # Limit frame rate and get time delta
            delta_ms: float = self.clock.tick(120)  # Target 120 FPS

            # Apply adaptive population control based on performance
            if self.frame_count % 10 == 0:
                if (
                    any(
                        ct.x.size > self.config.max_particles_per_type * 0.8
                        for ct in self.type_manager.cellular_types
                    )
                    or delta_ms > 16.67  # Below 60 FPS
                ):
                    self.cull_oldest_particles()

            # Terminate after the requested frame count
            if self.config.max_frames > 0 and self.frame_count >= self.config.max_frames:
                self.run_flag = False

        # Clean up Pygame resources on exit
        pygame.quit()

    @eidosian()
    def display_fps(self, surface: pygame.Surface, fps: float) -> None:
        """
        Display the current FPS counter in the top-left corner of the screen.

        Args:
            surface: Pygame surface to render the FPS text on
            fps: Current frames per second value
        """
        font: pygame.font.Font = pygame.font.Font(None, 36)
        fps_text: pygame.Surface = font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))
        surface.blit(fps_text, (10, 10))

    @eidosian()
    def apply_all_interactions(self) -> None:
        """
        Process all type-to-type interactions defined in the rules matrix.

        Iterates through all interaction rule parameters and applies them
        between the corresponding cellular type pairs using vectorized operations
        for maximum performance.
        """
        self._interaction_cache = self._build_interaction_cache()
        dvx: List[FloatArray] = [
            np.zeros_like(ct.vx, dtype=np.float64)
            for ct in self.type_manager.cellular_types
        ]
        dvy: List[FloatArray] = [
            np.zeros_like(ct.vy, dtype=np.float64)
            for ct in self.type_manager.cellular_types
        ]
        dvz: Optional[List[FloatArray]] = None
        if self.spatial_dimensions == 3:
            dvz = [
                np.zeros_like(ct.vz, dtype=np.float64)
                for ct in self.type_manager.cellular_types
            ]
        for i, j, params in self.rules_manager.rules:
            # Convert raw params to properly typed dictionary with validation
            typed_params: Dict[str, Union[float, bool, FloatArray]] = {}
            for k, v in params.items():
                if isinstance(v, bool):
                    typed_params[k] = v
                elif isinstance(v, (int, float)):
                    typed_params[k] = float(v)
                elif isinstance(v, np.ndarray):
                    # Check if array contains numeric data in a type-safe way
                    try:
                        # Direct check on dtype without hasattr
                        is_numeric = np.issubdtype(v.dtype, np.number)  # type: ignore
                        if is_numeric:
                            typed_params[k] = v
                    except (TypeError, AttributeError):
                        pass
                # Skip values that don't match our expected types
            self.apply_interaction_between_types(
                i,
                j,
                typed_params,
                dvx,
                dvy,
                dvz,
                self._interaction_cache,
            )

        # Integrate velocities/positions once per type per frame
        for idx, ct in enumerate(self.type_manager.cellular_types):
            if dvz is not None:
                self.integrate_type(ct, dvx[idx], dvy[idx], dvz[idx])
            else:
                self.integrate_type(ct, dvx[idx], dvy[idx])
        self._interaction_cache = None

    def _build_interaction_cache(
        self,
    ) -> Tuple[
        List[Optional[FloatArray]],
        List[Optional[KDTree]],
        List[Optional[IntArray]],
        Tuple[float, ...],
    ]:
        positions_cache: List[Optional[FloatArray]] = []
        tree_cache: List[Optional[KDTree]] = []
        index_map_cache: List[Optional[IntArray]] = []
        world_size = tuple(self.world_size.tolist())

        for ct in self.type_manager.cellular_types:
            if ct.x.size == 0:
                positions_cache.append(None)
                tree_cache.append(None)
                index_map_cache.append(None)
                continue

            if self.spatial_dimensions == 3:
                positions = np.column_stack((ct.x, ct.y, ct.z))
            else:
                positions = np.column_stack((ct.x, ct.y))

            if self.config.boundary_mode == "wrap":
                positions = positions.copy()
                positions[:, 0] = wrap_positions(positions[:, 0], 0.0, world_size[0])
                positions[:, 1] = wrap_positions(positions[:, 1], 0.0, world_size[1])
                if self.spatial_dimensions == 3:
                    positions[:, 2] = wrap_positions(
                        positions[:, 2], 0.0, world_size[2]
                    )

            positions_cache.append(positions)

            if self.config.boundary_mode == "wrap":
                try:
                    tree = KDTree(positions, boxsize=world_size)
                    index_map = None
                except TypeError:
                    tiled_positions, index_map = tile_positions_for_wrap(
                        positions, world_size
                    )
                    tree = KDTree(tiled_positions)
                    index_map = index_map.astype(np.int_, copy=False)
            else:
                tree = KDTree(positions)
                index_map = None

            tree_cache.append(tree)
            index_map_cache.append(index_map)

        return positions_cache, tree_cache, index_map_cache, world_size

    @eidosian()
    def apply_gene_interpreter(self) -> None:
        """Apply the genetic interpreter to all cellular types."""
        if not self.type_manager.cellular_types:
            return

        for idx, ct in enumerate(self.type_manager.cellular_types):
            others = [
                other
                for other_idx, other in enumerate(self.type_manager.cellular_types)
                if other_idx != idx
            ]
            self.genetic_interpreter.decode(ct, others, self.config)

    @eidosian()
    def apply_interaction_between_types(
        self,
        i: int,
        j: int,
        params: Dict[str, Union[float, bool, FloatArray]],
        dvx: List[FloatArray],
        dvy: List[FloatArray],
        dvz: Optional[List[FloatArray]] = None,
        interaction_cache: Optional[
            Tuple[
                List[Optional[FloatArray]],
                List[Optional[KDTree]],
                List[Optional[IntArray]],
                Tuple[float, ...],
            ]
        ] = None,
    ) -> None:
        """
        Apply physics, energy transfers, and synergy between two cellular types.

        Calculates forces between particles of different types based on distance,
        applies potential and gravitational forces, handles predator-prey energy
        transfers, and processes cooperative energy sharing (synergy).

        Args:
            i: Index of the first cellular type
            j: Index of the second cellular type
            params: Parameters controlling interaction physics and forces
        """
        # Retrieve cellular types and their interaction properties
        ct_i: CellularTypeData = self.type_manager.get_cellular_type_by_id(i)
        ct_j: CellularTypeData = self.type_manager.get_cellular_type_by_id(j)
        synergy_factor: float = float(self.rules_manager.synergy_matrix[i, j])
        is_giver: bool = bool(self.rules_manager.give_take_matrix[i, j])

        # Skip if either type has no particles
        n_i: int = ct_i.x.size
        n_j: int = ct_j.x.size
        if n_i == 0 or n_j == 0:
            return
        wrap_mode = self.config.boundary_mode == "wrap"
        world_size = tuple(self.world_size.tolist())
        inv_world_size: Optional[Tuple[float, ...]] = None
        if wrap_mode:
            if self.spatial_dimensions == 3:
                inv_world_size = (
                    1.0 / world_size[0],
                    1.0 / world_size[1],
                    1.0 / world_size[2],
                )
            else:
                inv_world_size = (1.0 / world_size[0], 1.0 / world_size[1])

        # Configure gravity parameters for mass-based types
        use_gravity = params.get("use_gravity", False)
        if isinstance(use_gravity, bool) and use_gravity:
            if (
                ct_i.mass_based
                and ct_i.mass is not None
                and ct_j.mass_based
                and ct_j.mass is not None
            ):
                params["m_a"] = ct_i.mass
                params["m_b"] = ct_j.mass
            else:
                params["use_gravity"] = False

        # Create interaction mask for particles within range
        max_dist_value = params.get("max_dist", 0.0)
        if isinstance(max_dist_value, np.ndarray):
            if max_dist_value.size == 1:
                max_dist = float(max_dist_value.item())
            else:
                max_dist = float(np.max(max_dist_value))
        else:
            max_dist = (
                float(max_dist_value)
                if not isinstance(max_dist_value, bool)
                else 100.0
            )
        if max_dist <= 0.0:
            return

        pair_count = n_i * n_j
        use_sparse = pair_count >= INTERACTION_KDTREE_THRESHOLD
        if use_sparse:
            if self.spatial_dimensions == 3:
                volume = float(np.prod(self.world_size))
                sphere = (4.0 / 3.0) * np.pi * (max_dist**3)
                dense_fraction = min(1.0, sphere / max(volume, 1.0))
            else:
                area = float(self.world_size[0] * self.world_size[1])
                circle = np.pi * (max_dist**2)
                dense_fraction = min(1.0, circle / max(area, 1.0))
            if dense_fraction >= INTERACTION_DENSE_FRACTION:
                use_sparse = False

        sparse_ready = False
        if use_sparse:
            if interaction_cache is not None:
                positions_cache, tree_cache, index_map_cache, _ = interaction_cache
                positions_i = positions_cache[i]
                positions_j = positions_cache[j]
                tree_i = tree_cache[i]
                tree_j = tree_cache[j]
                index_map_i = index_map_cache[i]
                index_map_j = index_map_cache[j]
            else:
                positions_i = None
                positions_j = None
                tree_i = None
                tree_j = None
                index_map_i = None
                index_map_j = None
                world_size = tuple(self.world_size.tolist())

            if positions_i is None or positions_j is None or tree_i is None or tree_j is None:
                use_sparse = False
            else:
                if index_map_i is None and index_map_j is None:
                    sparse_matrix = tree_i.sparse_distance_matrix(
                        tree_j, max_dist, output_type="coo_matrix"
                    )
                    if sparse_matrix.nnz == 0:
                        return
                    indices_i = sparse_matrix.row.astype(np.int_, copy=False)
                    indices_j = sparse_matrix.col.astype(np.int_, copy=False)
                    dist = sparse_matrix.data
                    valid_mask = dist > 0.0
                    if not np.any(valid_mask):
                        return
                    indices_i = indices_i[valid_mask]
                    indices_j = indices_j[valid_mask]
                    dist = dist[valid_mask]
                    dx = ct_i.x[indices_i] - ct_j.x[indices_j]
                    dy = ct_i.y[indices_i] - ct_j.y[indices_j]
                    if wrap_mode and inv_world_size is not None:
                        dx = wrap_deltas(dx, world_size[0], inv_world_size[0])
                        dy = wrap_deltas(dy, world_size[1], inv_world_size[1])
                    if self.spatial_dimensions == 3:
                        dz = ct_i.z[indices_i] - ct_j.z[indices_j]
                        if wrap_mode and inv_world_size is not None:
                            dz = wrap_deltas(dz, world_size[2], inv_world_size[2])
                    indices = (indices_i, indices_j)
                    sparse_ready = True
                else:
                    raw_neighbors = tree_j.query_ball_point(positions_i, max_dist)
                    if self.config.boundary_mode == "wrap" and index_map_j is not None:
                        neighbors_list = []
                        for idx, neighbors in enumerate(raw_neighbors):
                            if not neighbors:
                                neighbors_list.append([])
                                continue
                            mapped = index_map_j[np.asarray(neighbors, dtype=np.int_)]
                            if i == j:
                                mapped = mapped[mapped != idx]
                            if mapped.size == 0:
                                neighbors_list.append([])
                                continue
                            neighbors_list.append(np.unique(mapped).tolist())
                    else:
                        if i == j:
                            neighbors_list = [
                                [n for n in neighbors if n != idx]
                                for idx, neighbors in enumerate(raw_neighbors)
                            ]
                        else:
                            neighbors_list = raw_neighbors

                    counts = np.fromiter((len(n) for n in neighbors_list), dtype=np.int_)
                    if counts.sum() == 0:
                        return
                    indices_i = np.repeat(
                        np.arange(len(neighbors_list), dtype=np.int_), counts
                    )
                    indices_j = np.concatenate(
                        [np.asarray(n, dtype=np.int_) for n in neighbors_list if n]
                    )

                    dx = ct_i.x[indices_i] - ct_j.x[indices_j]
                    dy = ct_i.y[indices_i] - ct_j.y[indices_j]
                    if wrap_mode and inv_world_size is not None:
                        dx = wrap_deltas(dx, world_size[0], inv_world_size[0])
                        dy = wrap_deltas(dy, world_size[1], inv_world_size[1])
                    if self.spatial_dimensions == 3:
                        dz = ct_i.z[indices_i] - ct_j.z[indices_j]
                        if wrap_mode and inv_world_size is not None:
                            dz = wrap_deltas(dz, world_size[2], inv_world_size[2])
                        dist_sq = dx * dx + dy * dy + dz * dz
                    else:
                        dist_sq = dx * dx + dy * dy

                    within = dist_sq > 0.0
                    if not np.any(within):
                        return
                    indices = (
                        indices_i[within].astype(np.int_),
                        indices_j[within].astype(np.int_),
                    )
                    dist = np.sqrt(dist_sq[within])
                    dx = dx[within]
                    dy = dy[within]
                    if self.spatial_dimensions == 3:
                        dz = dz[within]  # type: ignore[assignment]
                    sparse_ready = True
        if not sparse_ready:
            # Calculate pairwise distances using full matrix operations
            dx: FloatArray = ct_i.x[:, np.newaxis] - ct_j.x
            dy: FloatArray = ct_i.y[:, np.newaxis] - ct_j.y
            if wrap_mode and inv_world_size is not None:
                dx = wrap_deltas(dx, world_size[0], inv_world_size[0])
                dy = wrap_deltas(dy, world_size[1], inv_world_size[1])
            if self.spatial_dimensions == 3:
                dz: FloatArray = ct_i.z[:, np.newaxis] - ct_j.z
                if wrap_mode and inv_world_size is not None:
                    dz = wrap_deltas(dz, world_size[2], inv_world_size[2])
                dist_sq: FloatArray = dx * dx + dy * dy + dz * dz
            else:
                dist_sq = dx * dx + dy * dy

            within_range: BoolArray = (dist_sq > 0.0) & (dist_sq <= max_dist**2)

            # Get indices of interacting particle pairs
            indices_tuple = np.where(within_range)
            if len(indices_tuple) != 2 or len(indices_tuple[0]) == 0:
                return

            indices = (
                indices_tuple[0].astype(np.int_),
                indices_tuple[1].astype(np.int_),
            )
            dist = np.sqrt(dist_sq[indices])

        # Resolve force model selection
        use_potential = params.get("use_potential", True)
        use_potential = bool(use_potential) if isinstance(use_potential, bool) else True
        use_gravity = bool(params.get("use_gravity", False))
        if not use_potential and not use_gravity and not is_giver and synergy_factor <= 0.0:
            return

        inv_dist: Optional[FloatArray] = None
        inv_dist_sq: Optional[FloatArray] = None
        if use_potential or use_gravity:
            inv_dist = 1.0 / dist
            inv_dist_sq = inv_dist * inv_dist

        fx: Optional[FloatArray] = None
        fy: Optional[FloatArray] = None
        fz: Optional[FloatArray] = None

        # Calculate potential-based forces
        if use_potential:
            pot_strength_value = params.get("potential_strength", 1.0)
            pot_strength = (
                float(pot_strength_value)
                if not isinstance(pot_strength_value, bool)
                else 1.0
            )
            pot_scale = pot_strength * inv_dist_sq  # type: ignore[operator]
            if use_sparse:
                fx = pot_scale * dx
                fy = pot_scale * dy
            else:
                fx = pot_scale * dx[indices]
                fy = pot_scale * dy[indices]
            if self.spatial_dimensions == 3:
                if use_sparse:
                    fz = pot_scale * dz
                else:
                    fz = pot_scale * dz[indices]

        # Calculate gravitational forces if applicable
        if use_gravity:
            gravity_factor_value = params.get("gravity_factor", 1.0)
            gravity_factor = (
                float(gravity_factor_value)
                if not isinstance(gravity_factor_value, bool)
                else 1.0
            )

            m_a = params.get("m_a")
            m_b = params.get("m_b")

            # Check that mass arrays are valid NumPy arrays
            if isinstance(m_a, np.ndarray) and isinstance(m_b, np.ndarray):
                # Calculate gravitational force between particles (G * m1 * m2 / r²)
                if use_sparse:
                    grav_scale = gravity_factor * inv_dist_sq * inv_dist  # type: ignore[operator]
                else:
                    grav_scale = gravity_factor / dist_sq[indices]
                F_grav = (m_a[indices[0]] * m_b[indices[1]] * grav_scale).astype(
                    np.float64
                )
                # Gravity pulls toward, not away from (negative direction)
                if use_sparse:
                    grav_fx = -F_grav * dx * inv_dist  # type: ignore[operator]
                    grav_fy = -F_grav * dy * inv_dist  # type: ignore[operator]
                else:
                    grav_fx = -F_grav * dx[indices] * inv_dist  # type: ignore[operator]
                    grav_fy = -F_grav * dy[indices] * inv_dist  # type: ignore[operator]
                if fx is None:
                    fx = grav_fx
                else:
                    fx += grav_fx
                if fy is None:
                    fy = grav_fy
                else:
                    fy += grav_fy
                if self.spatial_dimensions == 3:
                    if use_sparse:
                        grav_fz = -F_grav * dz * inv_dist  # type: ignore[operator]
                    else:
                        grav_fz = -F_grav * dz[indices] * inv_dist  # type: ignore[operator]
                    if fz is None:
                        fz = grav_fz
                    else:
                        fz += grav_fz

        # Accumulate forces for later integration
        if fx is not None and fx.size > 0:
            dvx[i] += np.bincount(indices[0], weights=fx, minlength=n_i)
            dvy[i] += np.bincount(indices[0], weights=fy, minlength=n_i)
            if fz is not None and dvz is not None:
                dvz[i] += np.bincount(indices[0], weights=fz, minlength=n_i)

        # Handle predator-prey energy transfers (give-take)
        if is_giver:
            # Find pairs within predation range
            if use_sparse:
                give_take_within = dist <= self.config.predation_range
            else:
                give_take_within = dist_sq[indices] <= self.config.predation_range**2
            give_take_indices: Tuple[IntArray, IntArray] = (
                indices[0][give_take_within],
                indices[1][give_take_within],
            )

            if give_take_indices[0].size > 0:
                # Extract energy and mass values for transfer
                giver_energy: FloatArray = ct_i.energy[give_take_indices[0]]
                receiver_energy: FloatArray = ct_j.energy[give_take_indices[1]]
                giver_mass: Optional[FloatArray] = (
                    ct_i.mass[give_take_indices[0]]
                    if ct_i.mass_based and ct_i.mass is not None
                    else None
                )
                receiver_mass: Optional[FloatArray] = (
                    ct_j.mass[give_take_indices[1]]
                    if ct_j.mass_based and ct_j.mass is not None
                    else None
                )

                # Process energy and mass transfers
                updated: Tuple[
                    FloatArray,
                    FloatArray,
                    Optional[FloatArray],
                    Optional[FloatArray],
                ] = give_take_interaction(
                    giver_energy,
                    receiver_energy,
                    giver_mass,
                    receiver_mass,
                    self.config,
                )

                # Update energy and mass values after transfer
                ct_i.energy[give_take_indices[0]] = updated[0]
                ct_j.energy[give_take_indices[1]] = updated[1]

                if ct_i.mass_based and ct_i.mass is not None and updated[2] is not None:
                    ct_i.mass[give_take_indices[0]] = updated[2]
                if ct_j.mass_based and ct_j.mass is not None and updated[3] is not None:
                    ct_j.mass[give_take_indices[1]] = updated[3]

        # Handle synergy (cooperative energy sharing)
        if synergy_factor > 0.0 and self.config.synergy_range > 0.0:
            # Find pairs within synergy range
            if use_sparse:
                synergy_within = dist <= self.config.synergy_range
            else:
                synergy_within = dist_sq[indices] <= self.config.synergy_range**2
            synergy_indices: Tuple[IntArray, IntArray] = (
                indices[0][synergy_within],
                indices[1][synergy_within],
            )

            if synergy_indices[0].size > 0:
                # Extract energy values for redistribution
                energyA: FloatArray = ct_i.energy[synergy_indices[0]]
                energyB: FloatArray = ct_j.energy[synergy_indices[1]]

                # Apply energy sharing based on synergy factor
                new_energyA, new_energyB = apply_synergy(
                    energyA, energyB, synergy_factor
                )
                ct_i.energy[synergy_indices[0]] = new_energyA
                ct_j.energy[synergy_indices[1]] = new_energyB

        # Integration and state updates are handled once per frame in integrate_type()

    @eidosian()
    def integrate_type(
        self,
        ct: CellularTypeData,
        dvx: FloatArray,
        dvy: FloatArray,
        dvz: Optional[FloatArray] = None,
    ) -> None:
        """Integrate velocity/position and update lifecycle once per frame."""
        if ct.x.size == 0:
            return

        ct.vx += dvx
        ct.vy += dvy
        if self.spatial_dimensions == 3:
            if dvz is None:
                dvz = np.zeros_like(ct.vz, dtype=np.float64)
            ct.vz += dvz

        # Apply friction to velocities
        friction_factor: float = 1.0 - self.config.friction
        ct.vx *= friction_factor
        ct.vy *= friction_factor
        if self.spatial_dimensions == 3:
            ct.vz *= friction_factor

        # Apply thermal noise (random motion)
        thermal_noise_x: FloatArray = (
            np.random.uniform(-0.5, 0.5, ct.x.size) * self.config.global_temperature
        )
        thermal_noise_y: FloatArray = (
            np.random.uniform(-0.5, 0.5, ct.x.size) * self.config.global_temperature
        )
        ct.vx += thermal_noise_x
        ct.vy += thermal_noise_y
        if self.spatial_dimensions == 3:
            thermal_noise_z: FloatArray = (
                np.random.uniform(-0.5, 0.5, ct.x.size) * self.config.global_temperature
            )
            ct.vz += thermal_noise_z

        # Update positions based on velocities
        ct.x += ct.vx
        ct.y += ct.vy
        if self.spatial_dimensions == 3:
            ct.z += ct.vz

        # Handle boundary conditions and state updates
        self.handle_boundary_reflections(ct)
        ct.age_components()
        ct.update_states()
        ct.update_alive()

    @eidosian()
    def handle_boundary_reflections(
        self, ct: Optional[CellularTypeData] = None
    ) -> None:
        """
        Reflect particles at screen boundaries and clamp positions to valid range.

        When particles reach screen edges, their velocities are reversed in the
        appropriate dimension and positions are constrained to remain within bounds.
        Uses vectorized operations for maximum performance.

        Args:
            ct: Specific cellular type to process; if None, processes all types
        """
        # Determine which cellular types to process
        cellular_types: List[CellularTypeData] = (
            [ct] if ct is not None else self.type_manager.cellular_types
        )

        for ct in cellular_types:
            if ct.x.size == 0:
                continue

            if self.config.boundary_mode == "wrap":
                ct.x = wrap_positions(ct.x, self.world_bounds[0], self.world_bounds[1])
                ct.y = wrap_positions(ct.y, self.world_bounds[2], self.world_bounds[3])
                if self.spatial_dimensions == 3 and self.depth_bounds is not None:
                    ct.z = wrap_positions(ct.z, self.depth_bounds[0], self.depth_bounds[1])
                continue

            # Create boolean masks for boundary violations in each direction
            left_mask: BoolArray = ct.x < self.screen_bounds[0]
            right_mask: BoolArray = ct.x > self.screen_bounds[1]
            top_mask: BoolArray = ct.y < self.screen_bounds[2]
            bottom_mask: BoolArray = ct.y > self.screen_bounds[3]

            # Reflect velocities for particles at boundaries
            ct.vx[left_mask | right_mask] *= -1
            ct.vy[top_mask | bottom_mask] *= -1

            # Clamp positions to remain within screen bounds
            np.clip(ct.x, self.screen_bounds[0], self.screen_bounds[1], out=ct.x)
            np.clip(ct.y, self.screen_bounds[2], self.screen_bounds[3], out=ct.y)
            if self.spatial_dimensions == 3 and self.depth_bounds is not None:
                near_mask: BoolArray = ct.z < self.depth_bounds[0]
                far_mask: BoolArray = ct.z > self.depth_bounds[1]
                ct.vz[near_mask | far_mask] *= -1
                np.clip(
                    ct.z, self.depth_bounds[0], self.depth_bounds[1], out=ct.z
                )

    @eidosian()
    def cull_oldest_particles(self) -> None:
        """
        Remove oldest particles from populated cellular types to maintain performance.

        When a cellular type exceeds a size threshold (500 particles), removes its
        oldest particle to prevent performance degradation. This helps maintain
        framerate while preserving the overall ecological balance.

        Uses vectorized operations to efficiently identify and remove old particles
        across all component arrays.
        """
        for ct in self.type_manager.cellular_types:
            # Skip types with reasonable population sizes
            if ct.x.size < 500:
                continue

            # Identify the oldest particle
            oldest_idx: int = int(np.argmax(ct.age))

            # Create a mask excluding the oldest particle
            keep_mask: BoolArray = np.ones(ct.x.size, dtype=bool)
            keep_mask[oldest_idx] = False

            ct.filter_by_mask(keep_mask)

    @eidosian()
    def add_global_energy(self) -> None:
        """
        Increase energy levels of all particles across the simulation.

        Adds 10% to current energy levels of all particles across all types,
        clamping to maximum allowed value of 200 units. Simulates environmental
        energy input into the system.

        Uses vectorized operations for efficient batch processing.
        """
        for ct in self.type_manager.cellular_types:
            # Add energy with bound checking (10% increase with ceiling)
            ct.energy = np.clip(ct.energy * 1.1, 0.0, self.config.max_energy)

    @eidosian()
    def apply_clustering(self, ct: CellularTypeData) -> None:
        """
        Apply flocking behavior within a cellular type using the Boids algorithm.

        Implements three steering behaviors:
        1. Alignment: Match velocity with nearby neighbors
        2. Cohesion: Move toward the center of nearby neighbors
        3. Separation: Avoid crowding nearby neighbors

        Uses KD-Tree for efficient nearest neighbor queries and vectorized
        operations for maximum performance.

        Args:
            ct: Cellular type to apply clustering behavior to
        """
        # Skip if too few particles for meaningful clustering
        n: int = ct.x.size
        if n < 2:
            return

        # Build KD-Tree for efficient neighbor searching
        if self.spatial_dimensions == 3:
            positions = np.column_stack((ct.x, ct.y, ct.z))
            world_size = tuple(self.world_size.tolist())
        else:
            positions = np.column_stack((ct.x, ct.y))
            world_size = tuple(self.world_size.tolist())

        index_map = None
        tree: Optional[KDTree] = None
        raw_neighbors = None
        if self.config.boundary_mode == "wrap":
            try:
                positions = positions.copy()
                positions[:, 0] = wrap_positions(positions[:, 0], 0.0, world_size[0])
                positions[:, 1] = wrap_positions(positions[:, 1], 0.0, world_size[1])
                if self.spatial_dimensions == 3:
                    positions[:, 2] = wrap_positions(positions[:, 2], 0.0, world_size[2])
                tree = KDTree(positions, boxsize=world_size)
            except TypeError:
                tiled_positions, index_map = tile_positions_for_wrap(positions, world_size)
                tree = KDTree(tiled_positions)
        else:
            tree = KDTree(positions)

        use_sparse = False
        if tree is not None and index_map is None and hasattr(tree, "sparse_distance_matrix"):
            try:
                sparse = tree.sparse_distance_matrix(
                    tree, self.config.cluster_radius, output_type="coo_matrix"
                )
                use_sparse = True
            except Exception:
                use_sparse = False

        if use_sparse:
            rows = sparse.row.astype(np.int_, copy=False)
            cols = sparse.col.astype(np.int_, copy=False)
            valid_mask = (rows != cols) & ct.alive[cols]
            if not np.any(valid_mask):
                return
            rows = rows[valid_mask]
            cols = cols[valid_mask]

            counts = np.bincount(rows, minlength=n).astype(np.float64)
            nonzero = counts > 0.0
            if not np.any(nonzero):
                return

            if self.spatial_dimensions == 3:
                vel = np.column_stack((ct.vx, ct.vy, ct.vz))
                sum_vel = np.column_stack(
                    (
                        np.bincount(rows, weights=vel[cols, 0], minlength=n),
                        np.bincount(rows, weights=vel[cols, 1], minlength=n),
                        np.bincount(rows, weights=vel[cols, 2], minlength=n),
                    )
                )
                sum_pos = np.column_stack(
                    (
                        np.bincount(rows, weights=positions[cols, 0], minlength=n),
                        np.bincount(rows, weights=positions[cols, 1], minlength=n),
                        np.bincount(rows, weights=positions[cols, 2], minlength=n),
                    )
                )
            else:
                vel = np.column_stack((ct.vx, ct.vy))
                sum_vel = np.column_stack(
                    (
                        np.bincount(rows, weights=vel[cols, 0], minlength=n),
                        np.bincount(rows, weights=vel[cols, 1], minlength=n),
                    )
                )
                sum_pos = np.column_stack(
                    (
                        np.bincount(rows, weights=positions[cols, 0], minlength=n),
                        np.bincount(rows, weights=positions[cols, 1], minlength=n),
                    )
                )

            inv_counts = np.zeros_like(counts)
            inv_counts[nonzero] = 1.0 / counts[nonzero]
            avg_vel = sum_vel * inv_counts[:, None]
            center = sum_pos * inv_counts[:, None]

            alignment = (avg_vel - vel) * self.config.alignment_strength
            cohesion = (center - positions) * self.config.cohesion_strength
            separation = (positions - center) * self.config.separation_strength
            total_force = alignment + cohesion + separation
            total_force[~nonzero] = 0.0

            ct.vx += total_force[:, 0]
            ct.vy += total_force[:, 1]
            if self.spatial_dimensions == 3:
                ct.vz += total_force[:, 2]
            return

        # Fallback to neighbor lists when sparse matrix is unavailable
        if tree is None:
            return

        raw_neighbors = tree.query_ball_point(positions, self.config.cluster_radius)
        if index_map is None:
            indices_list = [
                [i for i in neighbors if i != idx]
                for idx, neighbors in enumerate(raw_neighbors)
            ]
        else:
            indices_list = []
            for idx, neighbors in enumerate(raw_neighbors):
                mapped = [int(index_map[n]) for n in neighbors if int(index_map[n]) != idx]
                unique = list(dict.fromkeys(mapped))
                indices_list.append(unique)

        dvx: FloatArray = np.zeros(n, dtype=np.float64)
        dvy: FloatArray = np.zeros(n, dtype=np.float64)
        dvz: Optional[FloatArray] = np.zeros(n, dtype=np.float64) if self.spatial_dimensions == 3 else None

        for idx, neighbor_indices in enumerate(indices_list):
            filtered_indices: List[int] = [
                i for i in neighbor_indices if i != idx and ct.alive[i]
            ]
            if not filtered_indices:
                continue

            neighbor_positions: FloatArray = positions[filtered_indices]
            if self.spatial_dimensions == 3:
                neighbor_velocities = np.column_stack(
                    (
                        ct.vx[filtered_indices],
                        ct.vy[filtered_indices],
                        ct.vz[filtered_indices],
                    )
                )
                current_velocity = np.array(
                    [ct.vx[idx], ct.vy[idx], ct.vz[idx]], dtype=np.float64
                )
            else:
                neighbor_velocities = np.column_stack(
                    (ct.vx[filtered_indices], ct.vy[filtered_indices])
                )
                current_velocity = np.array([ct.vx[idx], ct.vy[idx]], dtype=np.float64)

            avg_velocity: FloatArray = np.mean(neighbor_velocities, axis=0)
            center: FloatArray = np.mean(neighbor_positions, axis=0)

            alignment: FloatArray = (avg_velocity - current_velocity) * self.config.alignment_strength
            cohesion: FloatArray = (center - positions[idx]) * self.config.cohesion_strength
            separation: FloatArray = (positions[idx] - center) * self.config.separation_strength

            total_force: FloatArray = alignment + cohesion + separation
            dvx[idx] = total_force[0]
            dvy[idx] = total_force[1]
            if dvz is not None:
                dvz[idx] = total_force[2]

        ct.vx += dvx
        ct.vy += dvy
        if dvz is not None:
            ct.vz += dvz
