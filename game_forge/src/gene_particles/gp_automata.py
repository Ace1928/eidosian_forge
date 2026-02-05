"""Gene Particles Cellular Automata Simulation.

Provides the core simulation framework for cellular automata with emergent evolution,
interaction physics, and environmental dynamics using vectorized operations for
maximum performance with precise static typing throughout.
"""

from collections import defaultdict
from dataclasses import dataclass
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


# Heuristic controls for interaction neighbor search (legacy fallback path)
INTERACTION_KDTREE_THRESHOLD = 50000
INTERACTION_DENSE_FRACTION = 0.35


@dataclass
class GlobalParticleView:
    """Global, contiguous view of all particles for batch interaction processing."""

    positions: FloatArray
    velocities: FloatArray
    energy: FloatArray
    mass: FloatArray
    alive: BoolArray
    type_ids: IntArray
    offsets: IntArray
    counts: IntArray
    total: int


@dataclass
class GlobalNeighborGraph:
    """Neighbor graph built over the global particle view."""

    rows: IntArray
    cols: IntArray
    dist: FloatArray
    wrap_mode: bool
    world_size: Tuple[float, ...]
    inv_world_size: Optional[Tuple[float, ...]]
    filtered_by_max_dist: bool = False


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
        self._global_capacity: int = 0
        self._global_positions: Optional[FloatArray] = None
        self._global_velocities: Optional[FloatArray] = None
        self._global_energy: Optional[FloatArray] = None
        self._global_mass: Optional[FloatArray] = None
        self._global_alive: Optional[BoolArray] = None
        self._global_type_ids: Optional[IntArray] = None

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

            # Apply clustering across all cellular types
            self.apply_clustering_all()

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
    def _ensure_global_buffers(self, total: int) -> None:
        """Ensure global particle buffers are allocated with sufficient capacity.

        Allocates contiguous arrays for positions, velocities, energy, mass,
        alive flags, and type ids. Capacity grows geometrically to minimize
        reallocations across frames.
        """
        if total <= self._global_capacity and self._global_positions is not None:
            return

        new_capacity = max(total, int(self._global_capacity * 1.5), 1024)
        dims = 3 if self.spatial_dimensions == 3 else 2
        self._global_positions = np.empty((new_capacity, dims), dtype=np.float64)
        self._global_velocities = np.empty((new_capacity, dims), dtype=np.float64)
        self._global_energy = np.empty(new_capacity, dtype=np.float64)
        self._global_mass = np.empty(new_capacity, dtype=np.float64)
        self._global_alive = np.empty(new_capacity, dtype=bool)
        self._global_type_ids = np.empty(new_capacity, dtype=np.int_)
        self._global_capacity = new_capacity

    def _build_global_view(self) -> Optional[GlobalParticleView]:
        """Create a contiguous view of all particles for batch processing.

        Builds a packed, global view of all particle state arrays and records
        per-type offsets/counts to allow zero-copy slicing back into type-local
        storage after batched interaction updates.
        """
        cellular_types = self.type_manager.cellular_types
        if not cellular_types:
            return None

        counts = np.fromiter((ct.x.size for ct in cellular_types), dtype=np.int_)
        total = int(counts.sum())
        if total == 0:
            return None

        offsets = np.empty(len(cellular_types) + 1, dtype=np.int_)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])

        self._ensure_global_buffers(total)
        assert self._global_positions is not None
        assert self._global_velocities is not None
        assert self._global_energy is not None
        assert self._global_mass is not None
        assert self._global_alive is not None
        assert self._global_type_ids is not None

        positions = self._global_positions[:total]
        velocities = self._global_velocities[:total]
        energy = self._global_energy[:total]
        mass = self._global_mass[:total]
        alive = self._global_alive[:total]
        type_ids = self._global_type_ids[:total]

        type_ids[:] = np.repeat(
            np.arange(len(cellular_types), dtype=np.int_), counts
        )

        for type_idx, ct in enumerate(cellular_types):
            start, end = offsets[type_idx], offsets[type_idx + 1]
            if start == end:
                continue
            positions[start:end, 0] = ct.x
            positions[start:end, 1] = ct.y
            velocities[start:end, 0] = ct.vx
            velocities[start:end, 1] = ct.vy
            if self.spatial_dimensions == 3:
                positions[start:end, 2] = ct.z
                velocities[start:end, 2] = ct.vz
            energy[start:end] = ct.energy
            alive[start:end] = ct.alive
            if ct.mass_based and ct.mass is not None:
                mass[start:end] = ct.mass
            else:
                mass[start:end] = 0.0

        return GlobalParticleView(
            positions=positions,
            velocities=velocities,
            energy=energy,
            mass=mass,
            alive=alive,
            type_ids=type_ids,
            offsets=offsets,
            counts=counts,
            total=total,
        )

    def _build_global_neighbor_graph(
        self, view: GlobalParticleView, max_dist: float
    ) -> Optional[GlobalNeighborGraph]:
        """Build a global neighbor graph for all particles within max_dist.

        Uses a single KDTree over the packed global positions to compute a
        sparse radius graph in COO form (rows, cols, dist). This path is used
        as a fallback when per-type graph construction is unavailable.
        """
        if max_dist <= 0.0 or view.total == 0:
            empty = np.array([], dtype=np.int_)
            empty_f = np.array([], dtype=np.float64)
            return GlobalNeighborGraph(
                rows=empty,
                cols=empty,
                dist=empty_f,
                wrap_mode=False,
                world_size=tuple(self.world_size.tolist()),
                inv_world_size=None,
                filtered_by_max_dist=False,
            )

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

        positions = view.positions
        positions_for_tree = positions
        if wrap_mode:
            positions_for_tree = positions.copy()
            positions_for_tree[:, 0] = wrap_positions(
                positions_for_tree[:, 0], 0.0, world_size[0]
            )
            positions_for_tree[:, 1] = wrap_positions(
                positions_for_tree[:, 1], 0.0, world_size[1]
            )
            if self.spatial_dimensions == 3:
                positions_for_tree[:, 2] = wrap_positions(
                    positions_for_tree[:, 2], 0.0, world_size[2]
                )

        try:
            if wrap_mode:
                tree = KDTree(positions_for_tree, boxsize=world_size)
            else:
                tree = KDTree(positions_for_tree)
        except TypeError:
            return None

        if not hasattr(tree, "sparse_distance_matrix"):
            return None

        try:
            sparse = tree.sparse_distance_matrix(tree, max_dist, output_type="coo_matrix")
        except Exception:
            return None
        if sparse.nnz == 0:
            empty = np.array([], dtype=np.int_)
            empty_f = np.array([], dtype=np.float64)
            return GlobalNeighborGraph(
                rows=empty,
                cols=empty,
                dist=empty_f,
                wrap_mode=wrap_mode,
                world_size=world_size,
                inv_world_size=inv_world_size,
                filtered_by_max_dist=False,
            )

        rows = sparse.row.astype(np.int_, copy=False)
        cols = sparse.col.astype(np.int_, copy=False)
        dist = sparse.data.astype(np.float64, copy=False)
        valid_mask = (rows != cols) & (dist > 0.0)
        if not np.any(valid_mask):
            empty = np.array([], dtype=np.int_)
            empty_f = np.array([], dtype=np.float64)
            return GlobalNeighborGraph(
                rows=empty,
                cols=empty,
                dist=empty_f,
                wrap_mode=wrap_mode,
                world_size=world_size,
                inv_world_size=inv_world_size,
                filtered_by_max_dist=False,
            )

        return GlobalNeighborGraph(
            rows=rows[valid_mask],
            cols=cols[valid_mask],
            dist=dist[valid_mask],
            wrap_mode=wrap_mode,
            world_size=world_size,
            inv_world_size=inv_world_size,
            filtered_by_max_dist=False,
        )

    def _build_global_neighbor_graph_pairwise(
        self, view: GlobalParticleView, max_dist_matrix: FloatArray
    ) -> Optional[GlobalNeighborGraph]:
        """Build a global neighbor graph by unioning per-pair radius searches.

        For each type pair (i, j), runs a radius search using per-type KDTree
        instances and the pair-specific max_dist from the interaction matrix.
        The resulting per-pair edges are concatenated into global (rows, cols, dist)
        arrays that drive batched interaction evaluation.
        """
        if view.total == 0:
            empty = np.array([], dtype=np.int_)
            empty_f = np.array([], dtype=np.float64)
            return GlobalNeighborGraph(
                rows=empty,
                cols=empty,
                dist=empty_f,
                wrap_mode=False,
                world_size=tuple(self.world_size.tolist()),
                inv_world_size=None,
                filtered_by_max_dist=True,
            )

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

        positions_cache, tree_cache, index_map_cache, _ = self._build_interaction_cache()
        if any(index_map_cache):
            return None

        rows_list: List[IntArray] = []
        cols_list: List[IntArray] = []
        dist_list: List[FloatArray] = []

        n_types = len(tree_cache)
        for i in range(n_types):
            tree_i = tree_cache[i]
            if tree_i is None:
                continue
            for j in range(n_types):
                tree_j = tree_cache[j]
                if tree_j is None:
                    continue
                max_dist = float(max_dist_matrix[i, j])
                if max_dist <= 0.0:
                    continue
                try:
                    sparse = tree_i.sparse_distance_matrix(
                        tree_j, max_dist, output_type="coo_matrix"
                    )
                except Exception:
                    return None
                if sparse.nnz == 0:
                    continue
                indices_i = sparse.row.astype(np.int_, copy=False)
                indices_j = sparse.col.astype(np.int_, copy=False)
                dist = sparse.data.astype(np.float64, copy=False)
                valid_mask = dist > 0.0
                if not np.any(valid_mask):
                    continue
                rows_list.append(view.offsets[i] + indices_i[valid_mask])
                cols_list.append(view.offsets[j] + indices_j[valid_mask])
                dist_list.append(dist[valid_mask])

        if not rows_list:
            empty = np.array([], dtype=np.int_)
            empty_f = np.array([], dtype=np.float64)
            return GlobalNeighborGraph(
                rows=empty,
                cols=empty,
                dist=empty_f,
                wrap_mode=wrap_mode,
                world_size=world_size,
                inv_world_size=inv_world_size,
                filtered_by_max_dist=True,
            )

        return GlobalNeighborGraph(
            rows=np.concatenate(rows_list),
            cols=np.concatenate(cols_list),
            dist=np.concatenate(dist_list),
            wrap_mode=wrap_mode,
            world_size=world_size,
            inv_world_size=inv_world_size,
            filtered_by_max_dist=True,
        )

    @eidosian()
    def apply_all_interactions(self) -> None:
        """
        Process all type-to-type interactions defined in the rules matrix.

        Iterates through all interaction rule parameters and applies them
        between the corresponding cellular type pairs using vectorized operations
        for maximum performance.
        """
        global_view = self._build_global_view()
        if global_view is None:
            return

        (
            max_dist_matrix,
            potential_strength,
            use_potential,
            use_gravity,
            gravity_factor,
        ) = self.rules_manager.to_matrices()
        max_dist_global = float(np.max(max_dist_matrix)) if max_dist_matrix.size else 0.0
        max_dist_global = max(
            max_dist_global,
            float(self.config.predation_range),
            float(self.config.synergy_range),
        )

        graph = self._build_global_neighbor_graph_pairwise(
            global_view, max_dist_matrix
        )
        if graph is None:
            graph = self._build_global_neighbor_graph(global_view, max_dist_global)
        if graph is None:
            self._apply_all_interactions_legacy()
            return

        dv = self._apply_global_interactions(
            global_view,
            graph,
            max_dist_matrix,
            potential_strength,
            use_potential,
            use_gravity,
            gravity_factor,
        )

        # Scatter energy updates back to per-type arrays before integration
        for type_idx, ct in enumerate(self.type_manager.cellular_types):
            start, end = global_view.offsets[type_idx], global_view.offsets[type_idx + 1]
            if start == end:
                continue
            ct.energy[:] = global_view.energy[start:end]
            if ct.mass_based and ct.mass is not None:
                ct.mass[:] = global_view.mass[start:end]

        # Integrate velocities/positions once per type per frame
        for type_idx, ct in enumerate(self.type_manager.cellular_types):
            start, end = global_view.offsets[type_idx], global_view.offsets[type_idx + 1]
            if start == end:
                continue
            if self.spatial_dimensions == 3:
                self.integrate_type(
                    ct,
                    dv[0][start:end],
                    dv[1][start:end],
                    dv[2][start:end],
                )
            else:
                self.integrate_type(ct, dv[0][start:end], dv[1][start:end])

    def _apply_all_interactions_legacy(self) -> None:
        """Legacy per-type interaction path used as a fallback."""
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

    def _apply_global_interactions(
        self,
        view: GlobalParticleView,
        graph: GlobalNeighborGraph,
        max_dist_matrix: FloatArray,
        potential_strength: FloatArray,
        use_potential: BoolArray,
        use_gravity: BoolArray,
        gravity_factor: FloatArray,
    ) -> Tuple[FloatArray, FloatArray, Optional[FloatArray]]:
        """Apply batched interactions using the global neighbor graph.

        Computes potential and gravity forces, predation transfers, and synergy
        exchanges across all edges in the global neighbor graph. Returns per-particle
        velocity deltas that are scattered back into per-type arrays for integration.
        """
        dvx = np.zeros(view.total, dtype=np.float64)
        dvy = np.zeros(view.total, dtype=np.float64)
        dvz: Optional[FloatArray] = (
            np.zeros(view.total, dtype=np.float64) if self.spatial_dimensions == 3 else None
        )

        if graph.rows.size == 0:
            return dvx, dvy, dvz

        mass_based_types = np.fromiter(
            (ct.mass_based and ct.mass is not None for ct in self.type_manager.cellular_types),
            dtype=bool,
        )

        rows = graph.rows
        cols = graph.cols
        dist = graph.dist

        type_i = view.type_ids[rows]
        type_j = view.type_ids[cols]
        if not graph.filtered_by_max_dist:
            max_dist = max_dist_matrix[type_i, type_j]
            within_mask = dist <= max_dist
            if not np.any(within_mask):
                return dvx, dvy, dvz

            rows = rows[within_mask]
            cols = cols[within_mask]
            dist = dist[within_mask]
            type_i = type_i[within_mask]
            type_j = type_j[within_mask]

        dx = view.positions[rows, 0] - view.positions[cols, 0]
        dy = view.positions[rows, 1] - view.positions[cols, 1]
        if graph.wrap_mode and graph.inv_world_size is not None:
            dx = wrap_deltas(dx, graph.world_size[0], graph.inv_world_size[0])
            dy = wrap_deltas(dy, graph.world_size[1], graph.inv_world_size[1])
        if self.spatial_dimensions == 3:
            dz = view.positions[rows, 2] - view.positions[cols, 2]
            if graph.wrap_mode and graph.inv_world_size is not None:
                dz = wrap_deltas(dz, graph.world_size[2], graph.inv_world_size[2])
        else:
            dz = None

        inv_dist = 1.0 / dist
        inv_dist_sq = inv_dist * inv_dist

        pot_scale = potential_strength[type_i, type_j] * inv_dist_sq
        pot_scale *= use_potential[type_i, type_j]
        fx = pot_scale * dx
        fy = pot_scale * dy
        if dz is not None:
            fz = pot_scale * dz
        else:
            fz = None

        grav_mask = use_gravity[type_i, type_j]
        if grav_mask.size:
            grav_mask = grav_mask & mass_based_types[type_i] & mass_based_types[type_j]
        if np.any(grav_mask):
            grav_scale = gravity_factor[type_i, type_j] * inv_dist_sq * inv_dist
            grav_scale *= grav_mask
            F_grav = view.mass[rows] * view.mass[cols] * grav_scale
            fx -= F_grav * dx * inv_dist
            fy -= F_grav * dy * inv_dist
            if fz is not None and dz is not None:
                fz -= F_grav * dz * inv_dist

        dvx += np.bincount(rows, weights=fx, minlength=view.total)
        dvy += np.bincount(rows, weights=fy, minlength=view.total)
        if dvz is not None and fz is not None:
            dvz += np.bincount(rows, weights=fz, minlength=view.total)

        give_take_matrix = self.rules_manager.give_take_matrix
        synergy_matrix = self.rules_manager.synergy_matrix
        if (self.config.predation_range > 0.0 or self.config.synergy_range > 0.0) and (
            np.any(give_take_matrix) or np.any(synergy_matrix > 0.0)
        ):
            n_types = len(self.type_manager.cellular_types)
            pair_ids = type_i * n_types + type_j
            for src in range(n_types):
                for dst in range(n_types):
                    pair_mask = pair_ids == (src * n_types + dst)
                    if not np.any(pair_mask):
                        continue

                    if give_take_matrix[src, dst] and self.config.predation_range > 0.0:
                        pred_mask = pair_mask & (dist <= self.config.predation_range)
                        if np.any(pred_mask):
                            giver_idx = rows[pred_mask]
                            receiver_idx = cols[pred_mask]
                            receiver_energy = view.energy[receiver_idx]
                            transfer = receiver_energy * self.config.energy_transfer_factor
                            view.energy[receiver_idx] = receiver_energy - transfer
                            view.energy[giver_idx] = view.energy[giver_idx] + transfer

                            if self.config.mass_transfer:
                                mass_mask = mass_based_types[type_i[pred_mask]] & mass_based_types[
                                    type_j[pred_mask]
                                ]
                                if np.any(mass_mask):
                                    giver_mass_idx = giver_idx[mass_mask]
                                    receiver_mass_idx = receiver_idx[mass_mask]
                                    receiver_mass = view.mass[receiver_mass_idx]
                                    mass_transfer = (
                                        receiver_mass * self.config.energy_transfer_factor
                                    )
                                    view.mass[receiver_mass_idx] = (
                                        receiver_mass - mass_transfer
                                    )
                                    view.mass[giver_mass_idx] = (
                                        view.mass[giver_mass_idx] + mass_transfer
                                    )

                    if self.config.synergy_range > 0.0:
                        synergy_factor = float(synergy_matrix[src, dst])
                        if synergy_factor > 0.0:
                            synergy_mask = pair_mask & (dist <= self.config.synergy_range)
                            if np.any(synergy_mask):
                                idx_a = rows[synergy_mask]
                                idx_b = cols[synergy_mask]
                                energy_a = view.energy[idx_a]
                                energy_b = view.energy[idx_b]
                                avg_energy = (energy_a + energy_b) * 0.5
                                view.energy[idx_a] = (
                                    energy_a * (1.0 - synergy_factor)
                                ) + (avg_energy * synergy_factor)
                                view.energy[idx_b] = (
                                    energy_b * (1.0 - synergy_factor)
                                ) + (avg_energy * synergy_factor)

        return dvx, dvy, dvz

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
        """Apply clustering to all types using the global neighbor graph."""
        _ = ct
        self.apply_clustering_all()

    def apply_clustering_all(self) -> None:
        """Apply clustering across all types using a global neighbor graph.

        Builds a radius-limited global neighbor graph, filters to same-type edges,
        and computes alignment/cohesion/separation in a batched, vectorized path.
        """
        if self.config.cluster_radius <= 0.0:
            return

        view = self._build_global_view()
        if view is None:
            return

        graph = self._build_global_neighbor_graph(view, float(self.config.cluster_radius))
        if graph is None:
            for ct in self.type_manager.cellular_types:
                self._apply_clustering_legacy(ct)
            return

        if graph.rows.size == 0:
            return

        rows = graph.rows
        cols = graph.cols
        same_type = view.type_ids[rows] == view.type_ids[cols]
        alive_neighbors = view.alive[cols]
        mask = same_type & alive_neighbors
        if not np.any(mask):
            return

        rows = rows[mask]
        cols = cols[mask]
        counts = np.bincount(rows, minlength=view.total).astype(np.float64)
        nonzero = counts > 0.0
        if not np.any(nonzero):
            return

        vel = view.velocities
        pos = view.positions
        if self.spatial_dimensions == 3:
            sum_vel = np.column_stack(
                (
                    np.bincount(rows, weights=vel[cols, 0], minlength=view.total),
                    np.bincount(rows, weights=vel[cols, 1], minlength=view.total),
                    np.bincount(rows, weights=vel[cols, 2], minlength=view.total),
                )
            )
            sum_pos = np.column_stack(
                (
                    np.bincount(rows, weights=pos[cols, 0], minlength=view.total),
                    np.bincount(rows, weights=pos[cols, 1], minlength=view.total),
                    np.bincount(rows, weights=pos[cols, 2], minlength=view.total),
                )
            )
        else:
            sum_vel = np.column_stack(
                (
                    np.bincount(rows, weights=vel[cols, 0], minlength=view.total),
                    np.bincount(rows, weights=vel[cols, 1], minlength=view.total),
                )
            )
            sum_pos = np.column_stack(
                (
                    np.bincount(rows, weights=pos[cols, 0], minlength=view.total),
                    np.bincount(rows, weights=pos[cols, 1], minlength=view.total),
                )
            )

        inv_counts = np.zeros_like(counts)
        inv_counts[nonzero] = 1.0 / counts[nonzero]
        avg_vel = sum_vel * inv_counts[:, None]
        center = sum_pos * inv_counts[:, None]

        alignment = (avg_vel - vel) * self.config.alignment_strength
        cohesion = (center - pos) * self.config.cohesion_strength
        separation = (pos - center) * self.config.separation_strength
        total_force = alignment + cohesion + separation
        total_force[~nonzero] = 0.0

        vel += total_force

        # Scatter updated velocities back to per-type arrays
        for type_idx, ct in enumerate(self.type_manager.cellular_types):
            start, end = view.offsets[type_idx], view.offsets[type_idx + 1]
            if start == end:
                continue
            ct.vx[:] = vel[start:end, 0]
            ct.vy[:] = vel[start:end, 1]
            if self.spatial_dimensions == 3:
                ct.vz[:] = vel[start:end, 2]

    def _apply_clustering_legacy(self, ct: CellularTypeData) -> None:
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
