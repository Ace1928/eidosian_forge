import pygame

from game_forge.src.gene_particles.gp_config import SimulationConfig
from game_forge.src.gene_particles.gp_ui import SimulationUI


def test_ui_handle_event_toggles_and_config():
    pygame.init()
    surface = pygame.Surface((300, 200))
    config = SimulationConfig()
    ui = SimulationUI(surface, config)

    ui.handle_event(pygame.event.Event(pygame.KEYUP, {"key": pygame.K_h}), object())
    assert ui.state.show_help is True

    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_SPACE}), object())
    assert ui.state.paused is True

    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_n}), object())
    assert ui.state.single_step is True

    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_h}), object())
    assert ui.state.show_help is False

    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_g}), object())
    assert ui.state.show_config is False

    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_s}), object())
    assert ui.state.show_stats is False

    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_b}), object())
    assert config.boundary_mode == "reflect"

    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_m}), object())
    assert config.projection_mode == "orthographic"

    size_before = config.particle_size
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_EQUALS}), object())
    assert config.particle_size > size_before

    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_MINUS}), object())
    assert config.particle_size <= size_before

    cluster_before = config.cluster_radius
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFTBRACKET}), object())
    assert config.cluster_radius < cluster_before

    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHTBRACKET}), object())
    assert config.cluster_radius >= cluster_before

    temp_before = config.global_temperature
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_COMMA}), object())
    assert config.global_temperature < temp_before

    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_PERIOD}), object())
    assert config.global_temperature >= temp_before

    registry_before = config.use_force_registry
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_f}), object())
    assert config.use_force_registry is not registry_before

    scale_before = config.force_registry_family_scale.get("yukawa", 0.0)
    ui.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_y}), object())
    assert config.force_registry_family_scale.get("yukawa", 0.0) != scale_before

    pygame.quit()


def test_ui_render_panels():
    pygame.init()
    surface = pygame.Surface((320, 240))
    config = SimulationConfig()
    ui = SimulationUI(surface, config)
    stats = {"fps": 60.0, "total_species": 2.0, "total_particles": 10.0}
    ui.render(stats)
    pygame.quit()
