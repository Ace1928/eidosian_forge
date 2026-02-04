import numpy as np
import pygame

from game_forge.src.gene_particles.gp_config import SimulationConfig
from game_forge.src.gene_particles.gp_renderer import Renderer
from game_forge.src.gene_particles.gp_types import CellularTypeData


def test_renderer_draw_and_render():
    pygame.init()
    screen = pygame.display.set_mode((100, 100))
    config = SimulationConfig()
    config.spatial_dimensions = 2
    renderer = Renderer(screen, config)

    ct = CellularTypeData(
        type_id=0,
        color=(100, 150, 200),
        n_particles=2,
        window_width=100,
        window_height=100,
        initial_energy=50.0,
        max_age=100.0,
        mass=None,
    )
    renderer.draw_cellular_type(ct)
    renderer.render({"fps": 60.0, "total_species": 1.0, "total_particles": 2.0})
    pygame.quit()


def test_renderer_draw_3d_projection():
    pygame.init()
    screen = pygame.display.set_mode((120, 120))
    config = SimulationConfig()
    config.spatial_dimensions = 3
    config.world_depth = 200.0
    config.projection_mode = "orthographic"
    renderer = Renderer(screen, config)

    ct = CellularTypeData(
        type_id=0,
        color=(100, 150, 200),
        n_particles=2,
        window_width=120,
        window_height=120,
        initial_energy=50.0,
        max_age=100.0,
        mass=None,
        window_depth=200,
        spatial_dimensions=3,
    )
    ct.z[:] = np.array([10.0, 150.0])
    renderer.draw_cellular_type(ct)
    renderer.render({"fps": 60.0, "total_species": 1.0, "total_particles": 2.0})
    pygame.quit()


def test_renderer_draw_3d_perspective():
    pygame.init()
    screen = pygame.display.set_mode((120, 120))
    config = SimulationConfig()
    config.spatial_dimensions = 3
    config.world_depth = 200.0
    config.projection_mode = "perspective"
    renderer = Renderer(screen, config)

    ct = CellularTypeData(
        type_id=0,
        color=(100, 150, 200),
        n_particles=2,
        window_width=120,
        window_height=120,
        initial_energy=50.0,
        max_age=100.0,
        mass=None,
        window_depth=200,
        spatial_dimensions=3,
    )
    ct.z[:] = np.array([10.0, 150.0])
    renderer.draw_cellular_type(ct)
    renderer.render({"fps": 60.0, "total_species": 1.0, "total_particles": 2.0})
    pygame.quit()
