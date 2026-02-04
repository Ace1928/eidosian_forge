import importlib.util
import sys
import types

import numpy as np
import pygame
import pytest

from game_forge.src.gene_particles.gp_automata import CellularAutomata
from game_forge.src.gene_particles.gp_config import ReproductionMode, SimulationConfig


def _make_config(n_cell_types=1, particles_per_type=3, dimensions: int = 2):
    config = SimulationConfig()
    config.n_cell_types = n_cell_types
    config.particles_per_type = particles_per_type
    config.max_particles_per_type = max(1000, particles_per_type)
    config.mass_based_fraction = 0.0
    config.spatial_dimensions = dimensions
    if dimensions == 3:
        config.world_depth = 120.0
    return config


def test_apply_all_interactions_and_integration():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=3)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(120, 120))
    ct = automata.type_manager.cellular_types[0]
    x_before = ct.x.copy()
    automata.apply_all_interactions()
    assert ct.x.size == x_before.size
    automata.apply_clustering(ct)
    pygame.quit()


def test_apply_all_interactions_3d_path():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=3, dimensions=3)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(120, 120))
    automata.apply_all_interactions()
    pygame.quit()


def test_apply_clustering_small_population():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=1)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    ct = automata.type_manager.cellular_types[0]
    automata.apply_clustering(ct)
    pygame.quit()


def test_apply_gene_interpreter_enabled():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=2)
    config.use_gene_interpreter = True
    config.reproduction_mode = ReproductionMode.GENES
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    ct = automata.type_manager.cellular_types[0]
    vx_before = ct.vx.copy()
    automata.apply_gene_interpreter()
    assert ct.vx.size == vx_before.size
    pygame.quit()


def test_apply_gene_interpreter_empty_types():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=1)
    config.use_gene_interpreter = True
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    automata.type_manager.cellular_types = []
    automata.apply_gene_interpreter()
    pygame.quit()


def test_main_loop_runs_with_gene_interpreter():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=2)
    config.max_frames = 1
    config.use_gene_interpreter = True
    config.reproduction_mode = ReproductionMode.GENES
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    automata.main_loop()
    assert automata.frame_count == 1
    pygame.quit()


def test_apply_all_interactions_param_array():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=2)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    automata.rules_manager.rules = [(0, 0, {"max_dist": np.array([10.0])})]
    automata.apply_all_interactions()
    pygame.quit()


def test_apply_all_interactions_param_array_multi():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=2)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    automata.rules_manager.rules = [(0, 0, {"max_dist": np.array([10.0, 12.0])})]
    automata.apply_all_interactions()
    pygame.quit()


def test_apply_all_interactions_dtype_exception(monkeypatch):
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=2)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    automata.rules_manager.rules = [(0, 0, {"dummy": np.array([1.0])})]
    monkeypatch.setattr(np, "issubdtype", lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("boom")))
    automata.apply_all_interactions()
    pygame.quit()


def test_handle_boundary_reflections():
    pygame.init()
    config = _make_config()
    automata = CellularAutomata(config, fullscreen=False, screen_size=(50, 50))
    ct = automata.type_manager.cellular_types[0]
    ct.x[:] = -10.0
    ct.vx[:] = 1.0
    automata.handle_boundary_reflections(ct)
    assert (ct.vx <= 0).all()
    pygame.quit()


def test_handle_boundary_reflections_all_types_and_empty():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=1)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(50, 50))
    ct = automata.type_manager.cellular_types[0]
    ct.x = np.array([], dtype=np.float64)
    ct.y = np.array([], dtype=np.float64)
    ct.vx = np.array([], dtype=np.float64)
    ct.vy = np.array([], dtype=np.float64)
    automata.handle_boundary_reflections()
    pygame.quit()


def test_cull_oldest_particles():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=501)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(50, 50))
    ct = automata.type_manager.cellular_types[0]
    ct.age[:] = 1.0
    before = ct.x.size
    automata.cull_oldest_particles()
    assert ct.x.size == before - 1
    pygame.quit()


def test_main_loop_runs_once():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=2)
    config.max_frames = 1
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    automata.main_loop()
    assert automata.frame_count == 1
    pygame.quit()


def test_main_loop_quit_event():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=2)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    automata.main_loop()
    assert automata.run_flag is False
    pygame.quit()


def test_main_loop_cull_branch():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=9)
    config.max_particles_per_type = 10
    config.max_frames = 10
    automata = CellularAutomata(config, fullscreen=False, screen_size=(60, 60))
    automata.main_loop()
    assert automata.frame_count == 10
    pygame.quit()


def test_display_fps():
    pygame.init()
    config = _make_config()
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    automata.display_fps(automata.screen, 60.0)
    pygame.quit()


def test_add_global_energy():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=2)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    ct = automata.type_manager.cellular_types[0]
    energy_before = ct.energy.copy()
    automata.add_global_energy()
    assert (ct.energy >= energy_before).all()
    pygame.quit()


def test_integrate_type_empty():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=1)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(50, 50))
    ct = automata.type_manager.cellular_types[0]
    ct.x = np.array([], dtype=np.float64)
    ct.y = np.array([], dtype=np.float64)
    ct.vx = np.array([], dtype=np.float64)
    ct.vy = np.array([], dtype=np.float64)
    automata.integrate_type(ct, np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    pygame.quit()


def test_apply_interaction_between_types_branches():
    pygame.init()
    config = _make_config(n_cell_types=2, particles_per_type=2)
    config.predation_range = 50.0
    config.synergy_range = 50.0
    config.mass_based_fraction = 1.0
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    ct_a = automata.type_manager.cellular_types[0]
    ct_b = automata.type_manager.cellular_types[1]
    ct_a.x[:] = 0.0
    ct_a.y[:] = 0.0
    ct_b.x[:] = 1.0
    ct_b.y[:] = 1.0

    automata.rules_manager.synergy_matrix[0, 1] = 0.5
    automata.rules_manager.give_take_matrix[0, 1] = True

    dvx = [np.zeros_like(ct_a.vx), np.zeros_like(ct_b.vx)]
    dvy = [np.zeros_like(ct_a.vy), np.zeros_like(ct_b.vy)]
    params = {"use_gravity": True, "use_potential": True, "potential_strength": 1.0, "max_dist": 10.0}
    automata.apply_interaction_between_types(0, 1, params, dvx, dvy)
    automata.integrate_type(ct_a, dvx[0], dvy[0])
    automata.integrate_type(ct_b, dvx[1], dvy[1])
    pygame.quit()


def test_apply_interaction_between_types_3d_forces():
    pygame.init()
    config = _make_config(n_cell_types=2, particles_per_type=2, dimensions=3)
    config.predation_range = 50.0
    config.synergy_range = 50.0
    config.mass_based_fraction = 1.0
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    ct_a = automata.type_manager.cellular_types[0]
    ct_b = automata.type_manager.cellular_types[1]
    ct_a.x[:] = 0.0
    ct_a.y[:] = 0.0
    ct_a.z[:] = 0.0
    ct_b.x[:] = 1.0
    ct_b.y[:] = 1.0
    ct_b.z[:] = 2.0

    dvx = [np.zeros_like(ct_a.vx), np.zeros_like(ct_b.vx)]
    dvy = [np.zeros_like(ct_a.vy), np.zeros_like(ct_b.vy)]
    dvz = [np.zeros_like(ct_a.vz), np.zeros_like(ct_b.vz)]
    params = {"use_gravity": True, "use_potential": True, "potential_strength": 1.0, "max_dist": 10.0}
    automata.apply_interaction_between_types(0, 1, params, dvx, dvy, dvz)
    automata.integrate_type(ct_a, dvx[0], dvy[0], dvz[0])
    automata.integrate_type(ct_b, dvx[1], dvy[1], dvz[1])
    pygame.quit()


def test_apply_interaction_disable_gravity():
    pygame.init()
    config = _make_config(n_cell_types=2, particles_per_type=1)
    config.mass_based_fraction = 0.0
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    dvx = [np.zeros_like(automata.type_manager.cellular_types[0].vx), np.zeros_like(automata.type_manager.cellular_types[1].vx)]
    dvy = [np.zeros_like(automata.type_manager.cellular_types[0].vy), np.zeros_like(automata.type_manager.cellular_types[1].vy)]
    params = {"use_gravity": True, "use_potential": True, "potential_strength": 1.0, "max_dist": 10.0}
    automata.apply_interaction_between_types(0, 1, params, dvx, dvy)
    pygame.quit()


def test_apply_interaction_between_types_no_pairs():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=1)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    ct = automata.type_manager.cellular_types[0]
    ct.x[:] = 0.0
    ct.y[:] = 0.0
    dvx = [np.zeros_like(ct.vx)]
    dvy = [np.zeros_like(ct.vy)]
    automata.apply_interaction_between_types(0, 0, {"max_dist": 0.0}, dvx, dvy)
    pygame.quit()


def test_apply_interaction_between_types_empty_type():
    pygame.init()
    config = _make_config(n_cell_types=2, particles_per_type=1)
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    ct = automata.type_manager.cellular_types[0]
    ct.x = np.array([], dtype=np.float64)
    ct.y = np.array([], dtype=np.float64)
    ct.vx = np.array([], dtype=np.float64)
    ct.vy = np.array([], dtype=np.float64)
    dvx = [np.array([], dtype=np.float64), np.zeros_like(automata.type_manager.cellular_types[1].vx)]
    dvy = [np.array([], dtype=np.float64), np.zeros_like(automata.type_manager.cellular_types[1].vy)]
    automata.apply_interaction_between_types(0, 1, {"max_dist": 10.0}, dvx, dvy)
    pygame.quit()


def test_screen_size_default_path():
    pygame.init()
    config = _make_config()
    automata = CellularAutomata(config, fullscreen=False, screen_size=None)
    assert automata.screen.get_width() > 0
    pygame.quit()


def test_kdtree_fallback(monkeypatch):
    module_path = "game_forge/src/gene_particles/gp_automata.py"
    dummy = types.ModuleType("scipy")
    monkeypatch.setitem(sys.modules, "scipy", dummy)
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    spec = importlib.util.spec_from_file_location("gp_automata_stub", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    tree = module.KDTree(np.zeros((1, 2)))
    assert tree.query_ball_point(np.zeros((1, 2)), 1.0) == [[]]


def test_integrate_type_3d_and_boundaries():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=2, dimensions=3)
    config.global_temperature = 0.0
    config.friction = 0.0
    automata = CellularAutomata(config, fullscreen=False, screen_size=(60, 60))
    ct = automata.type_manager.cellular_types[0]
    ct.z[:] = -10.0
    ct.vz[:] = 1.0
    automata.integrate_type(ct, np.zeros_like(ct.vx), np.zeros_like(ct.vy))
    assert (ct.vz <= 0).all()
    pygame.quit()


def test_apply_clustering_3d_path():
    pygame.init()
    config = _make_config(n_cell_types=1, particles_per_type=2, dimensions=3)
    config.cluster_radius = 1000.0
    automata = CellularAutomata(config, fullscreen=False, screen_size=(80, 80))
    ct = automata.type_manager.cellular_types[0]
    ct.x[:] = np.array([10.0, 12.0])
    ct.y[:] = np.array([10.0, 12.0])
    ct.z[:] = np.array([10.0, 12.0])
    ct.vx[:] = 0.0
    ct.vy[:] = 0.0
    ct.vz[:] = 0.0
    automata.apply_clustering(ct)
    pygame.quit()
