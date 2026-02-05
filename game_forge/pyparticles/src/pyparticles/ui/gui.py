"""
Eidosian PyParticles V6 - Advanced GUI
Full-featured control panel with aesthetic design.
"""
import pygame
import pygame_gui
import json
import numpy as np
from pygame_gui.elements import (
    UIWindow, UIHorizontalSlider, UILabel, UIButton, UIPanel, 
    UIDropDownMenu, UITextEntryLine, UISelectionList, UIProgressBar
)
from pygame_gui.windows import UIFileDialog
from ..core.types import SimulationConfig, InteractionRule, ForceType, RenderMode


class SimulationGUI:
    """
    Advanced GUI for PyParticles V6.
    
    Features:
    - Particle size/count controls
    - Physics parameter sliders
    - Rule matrix editor
    - Species wave configuration
    - Visual presets
    - Performance monitoring
    - Save/Load configurations
    """
    
    def __init__(self, manager: pygame_gui.UIManager, config: SimulationConfig, physics_engine):
        self.manager = manager
        self.cfg = config
        self.physics = physics_engine
        self.window_size = (config.width, config.height)
        
        self.paused = False
        self.active_rule_idx = 0
        self.active_species_idx = 0
        self.file_dialog = None
        
        # Performance tracking
        self.fps_history = []
        self.step_times = []
        
        self._setup_main_panel()
        self._setup_visual_panel()
        self._setup_matrix_editor()
        self._setup_species_editor()
        self._setup_performance_panel()

    def _setup_main_panel(self):
        """Main control panel - left side."""
        rect = pygame.Rect(10, 10, 280, 750)
        self.hud_panel = UIPanel(
            relative_rect=rect,
            starting_height=1,
            manager=self.manager
        )
        
        y = 10
        
        # Title
        UILabel(pygame.Rect(10, y, 260, 30), "EIDOSIAN CONTROLS V6", 
                manager=self.manager, container=self.hud_panel)
        y += 35
        
        # Status labels
        self.fps_label = UILabel(pygame.Rect(10, y, 260, 20), "FPS: --", 
                                 manager=self.manager, container=self.hud_panel)
        y += 22
        self.count_label = UILabel(pygame.Rect(10, y, 260, 20), 
                                   f"Particles: {self.cfg.num_particles}",
                                   manager=self.manager, container=self.hud_panel)
        y += 22
        self.energy_label = UILabel(pygame.Rect(10, y, 260, 20), "Energy: --",
                                    manager=self.manager, container=self.hud_panel)
        y += 30
        
        # === PARTICLE CONFIGURATION ===
        self._add_section_header(y, "PARTICLES")
        y += 25
        
        # Particle count
        UILabel(pygame.Rect(10, y, 140, 20), "Count:", 
                manager=self.manager, container=self.hud_panel)
        self.input_particles = UITextEntryLine(
            pygame.Rect(90, y, 100, 25),
            manager=self.manager,
            container=self.hud_panel
        )
        self.input_particles.set_text(str(self.cfg.num_particles))
        self.btn_set_particles = UIButton(
            pygame.Rect(200, y, 60, 25), "Set",
            manager=self.manager, container=self.hud_panel
        )
        y += 30
        
        # Species count
        UILabel(pygame.Rect(10, y, 140, 20), "Species:", 
                manager=self.manager, container=self.hud_panel)
        self.input_species = UITextEntryLine(
            pygame.Rect(90, y, 100, 25),
            manager=self.manager,
            container=self.hud_panel
        )
        self.input_species.set_text(str(self.cfg.num_types))
        self.btn_set_species = UIButton(
            pygame.Rect(200, y, 60, 25), "Reset",
            manager=self.manager, container=self.hud_panel
        )
        y += 35
        
        # PARTICLE SIZE SLIDER
        UILabel(pygame.Rect(10, y, 180, 20), "Particle Size Scale:", 
                manager=self.manager, container=self.hud_panel)
        self.size_label = UILabel(pygame.Rect(190, y, 60, 20), "1.0x",
                                  manager=self.manager, container=self.hud_panel)
        y += 22
        self.size_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 250, 22),
            start_value=1.0,
            value_range=(0.1, 5.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 30
        
        # === PHYSICS PARAMETERS ===
        self._add_section_header(y, "PHYSICS")
        y += 25
        
        # Friction
        UILabel(pygame.Rect(10, y, 140, 20), "Friction:", 
                manager=self.manager, container=self.hud_panel)
        self.friction_value = UILabel(pygame.Rect(200, y, 60, 20), 
                                      f"{self.cfg.friction:.2f}",
                                      manager=self.manager, container=self.hud_panel)
        y += 22
        self.friction_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 250, 22),
            start_value=self.cfg.friction,
            value_range=(0.0, 2.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 28
        
        # Time Step
        UILabel(pygame.Rect(10, y, 140, 20), "Time Step:", 
                manager=self.manager, container=self.hud_panel)
        self.dt_value = UILabel(pygame.Rect(200, y, 60, 20), 
                                f"{self.cfg.dt:.4f}",
                                manager=self.manager, container=self.hud_panel)
        y += 22
        self.dt_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 250, 22),
            start_value=self.cfg.dt,
            value_range=(0.001, 0.02),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 28
        
        # Max Velocity
        UILabel(pygame.Rect(10, y, 140, 20), "Max Velocity:", 
                manager=self.manager, container=self.hud_panel)
        self.maxvel_value = UILabel(pygame.Rect(200, y, 60, 20), 
                                    f"{self.cfg.max_velocity:.1f}",
                                    manager=self.manager, container=self.hud_panel)
        y += 22
        self.maxvel_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 250, 22),
            start_value=self.cfg.max_velocity,
            value_range=(1.0, 50.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 35
        
        # === THERMOSTAT ===
        self._add_section_header(y, "THERMOSTAT")
        y += 25
        
        # Target Temperature
        UILabel(pygame.Rect(10, y, 140, 20), "Target Temp:", 
                manager=self.manager, container=self.hud_panel)
        self.temp_value = UILabel(pygame.Rect(200, y, 60, 20), 
                                  f"{self.cfg.target_temperature:.2f}",
                                  manager=self.manager, container=self.hud_panel)
        y += 22
        self.temp_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 250, 22),
            start_value=self.cfg.target_temperature,
            value_range=(0.0, 3.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 28
        
        # Thermostat Coupling
        UILabel(pygame.Rect(10, y, 140, 20), "Coupling:", 
                manager=self.manager, container=self.hud_panel)
        self.coupling_value = UILabel(pygame.Rect(200, y, 60, 20), 
                                      f"{self.cfg.thermostat_coupling:.2f}",
                                      manager=self.manager, container=self.hud_panel)
        y += 22
        self.coupling_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 250, 22),
            start_value=self.cfg.thermostat_coupling,
            value_range=(0.0, 0.5),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 35
        
        # === FORCE RULES ===
        self._add_section_header(y, "FORCE RULES")
        y += 25
        
        self.rule_list_box = UISelectionList(
            relative_rect=pygame.Rect(10, y, 250, 80),
            item_list=[],
            manager=self.manager,
            container=self.hud_panel,
            allow_multi_select=True
        )
        self._update_rule_list()
        y += 85
        
        # Rule selector dropdown
        rule_names = [r.name for r in self.physics.rules]
        self.rule_selector = UIDropDownMenu(
            options_list=rule_names if rule_names else ["No Rules"],
            starting_option=rule_names[0] if rule_names else "No Rules",
            relative_rect=pygame.Rect(10, y, 250, 28),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 32
        
        # Rule strength
        UILabel(pygame.Rect(10, y, 100, 20), "Strength:", 
                manager=self.manager, container=self.hud_panel)
        self.strength_value = UILabel(pygame.Rect(200, y, 60, 20), "1.0",
                                      manager=self.manager, container=self.hud_panel)
        y += 22
        self.strength_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 250, 22),
            start_value=self.physics.rules[0].strength if self.physics.rules else 1.0,
            value_range=(0.0, 10.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 28
        
        # Rule radius
        UILabel(pygame.Rect(10, y, 100, 20), "Radius:", 
                manager=self.manager, container=self.hud_panel)
        self.radius_value = UILabel(pygame.Rect(200, y, 60, 20), "0.3",
                                    manager=self.manager, container=self.hud_panel)
        y += 22
        self.radius_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 250, 22),
            start_value=self.physics.rules[0].max_radius if self.physics.rules else 0.3,
            value_range=(0.01, 5.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 35
        
        # === CONTROL BUTTONS ===
        self.pause_btn = UIButton(pygame.Rect(10, y, 120, 32), "Pause", 
                                  manager=self.manager, container=self.hud_panel)
        self.reset_btn = UIButton(pygame.Rect(140, y, 120, 32), "Reset Sim",
                                  manager=self.manager, container=self.hud_panel)

    def _setup_visual_panel(self):
        """Visual/rendering controls - second panel."""
        rect = pygame.Rect(10, 770, 280, 220)
        self.visual_panel = UIPanel(
            relative_rect=rect,
            starting_height=1,
            manager=self.manager
        )
        
        y = 10
        self._add_section_header_to(y, "VISUAL SETTINGS", self.visual_panel)
        y += 25
        
        # World Size
        UILabel(pygame.Rect(10, y, 140, 20), "World Size:", 
                manager=self.manager, container=self.visual_panel)
        self.worldsize_value = UILabel(pygame.Rect(200, y, 60, 20), 
                                       f"{self.cfg.world_size:.0f}",
                                       manager=self.manager, container=self.visual_panel)
        y += 22
        self.worldsize_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 250, 22),
            start_value=max(2.0, min(500.0, self.cfg.world_size)),  # Clamp to range
            value_range=(2.0, 500.0),  # Support classic world_size=2.0
            manager=self.manager,
            container=self.visual_panel
        )
        y += 30
        
        # Render mode selector
        UILabel(pygame.Rect(10, y, 100, 20), "Render Mode:", 
                manager=self.manager, container=self.visual_panel)
        y += 22
        self.render_mode_selector = UIDropDownMenu(
            options_list=["Standard", "Wave", "Energy", "Minimal"],
            starting_option="Standard",
            relative_rect=pygame.Rect(10, y, 250, 28),
            manager=self.manager,
            container=self.visual_panel
        )
        y += 35
        
        # Visual presets
        UILabel(pygame.Rect(10, y, 100, 20), "Preset:", 
                manager=self.manager, container=self.visual_panel)
        y += 22
        self.preset_selector = UIDropDownMenu(
            options_list=["Default", "Small", "Large", "Huge", "Classic", "Emergence", "Dense"],
            starting_option="Default",
            relative_rect=pygame.Rect(10, y, 250, 28),
            manager=self.manager,
            container=self.visual_panel
        )
        y += 35
        
        # Save/Load buttons
        self.save_btn = UIButton(pygame.Rect(10, y, 120, 30), "Save Config",
                                 manager=self.manager, container=self.visual_panel)
        self.load_btn = UIButton(pygame.Rect(140, y, 120, 30), "Load Config",
                                 manager=self.manager, container=self.visual_panel)

    def _setup_matrix_editor(self):
        """Matrix editing window."""
        cell_size = 28
        n_types = min(self.cfg.num_types, 12)  # Cap display at 12x12
        grid_size = n_types * cell_size + 60
        rect = pygame.Rect(self.cfg.width - grid_size - 10, 10, grid_size, grid_size + 50)
        
        self.matrix_window = UIWindow(
            rect=rect,
            manager=self.manager,
            window_display_title="Interaction Matrix",
            resizable=True
        )
        
        self.matrix_buttons = {} 
        self._rebuild_matrix_grid()
        
    def _rebuild_matrix_grid(self):
        """Rebuild the matrix button grid."""
        for btn in self.matrix_buttons.values():
            btn.kill()
        self.matrix_buttons = {}
        
        cell_size = 28
        n_types = min(self.cfg.num_types, 12)
        
        for r in range(n_types):
            for c in range(n_types):
                if r < self.physics.rules[0].matrix.shape[0] and c < self.physics.rules[0].matrix.shape[1]:
                    val = self.physics.rules[self.active_rule_idx].matrix[r, c]
                    btn_rect = pygame.Rect(10 + c*cell_size, 10 + r*cell_size, cell_size-2, cell_size-2)
                    
                    # Color code the value
                    if val > 0.3:
                        color = '#44AA44'  # Green (attract)
                    elif val < -0.3:
                        color = '#AA4444'  # Red (repel)
                    else:
                        color = '#666666'  # Gray (neutral)
                    
                    btn = UIButton(
                        relative_rect=btn_rect,
                        text=f"{val:.1f}",
                        manager=self.manager,
                        container=self.matrix_window
                    )
                    self.matrix_buttons[(r, c)] = btn

    def _setup_species_editor(self):
        """Species wave configuration window."""
        w = 520
        h = 180
        rect = pygame.Rect((self.cfg.width - w)//2, self.cfg.height - h - 10, w, h)
        
        self.species_window = UIWindow(
            rect=rect,
            manager=self.manager,
            window_display_title="Species Configuration",
            resizable=False
        )
        
        x = 10
        y = 10
        
        # Type selector
        UILabel(pygame.Rect(x, y, 100, 25), "Species Type:", 
                manager=self.manager, container=self.species_window)
        
        type_options = [f"Type {i}" for i in range(self.cfg.num_types)]
        self.species_selector = UIDropDownMenu(
            options_list=type_options,
            starting_option="Type 0",
            relative_rect=pygame.Rect(x+110, y, 120, 28),
            manager=self.manager,
            container=self.species_window
        )
        
        # Randomize button
        self.randomize_species_btn = UIButton(
            pygame.Rect(x + 250, y, 100, 28), "Randomize",
            manager=self.manager, container=self.species_window
        )
        
        # All species button
        self.randomize_all_btn = UIButton(
            pygame.Rect(x + 360, y, 100, 28), "Random All",
            manager=self.manager, container=self.species_window
        )
        
        y += 40
        
        # Row 1: Radius and Wave Freq
        UILabel(pygame.Rect(x, y, 80, 20), "Radius:", 
                manager=self.manager, container=self.species_window)
        self.species_radius_slider = UIHorizontalSlider(
            pygame.Rect(x+80, y, 150, 22),
            start_value=self.physics.species_config.radius[0],
            value_range=(0.01, 1.0),
            manager=self.manager,
            container=self.species_window
        )
        
        x2 = 260
        UILabel(pygame.Rect(x2, y, 80, 20), "Wave Freq:", 
                manager=self.manager, container=self.species_window)
        self.freq_slider = UIHorizontalSlider(
            pygame.Rect(x2+80, y, 150, 22),
            start_value=self.physics.species_config.wave_freq[0],
            value_range=(1.0, 10.0),
            manager=self.manager,
            container=self.species_window
        )
        y += 30
        
        # Row 2: Wave Amp and Phase Speed
        UILabel(pygame.Rect(x, y, 80, 20), "Wave Amp:", 
                manager=self.manager, container=self.species_window)
        self.amp_slider = UIHorizontalSlider(
            pygame.Rect(x+80, y, 150, 22),
            start_value=self.physics.species_config.wave_amp[0],
            value_range=(0.0, 0.2),
            manager=self.manager,
            container=self.species_window
        )
        
        UILabel(pygame.Rect(x2, y, 80, 20), "Phase Spd:", 
                manager=self.manager, container=self.species_window)
        self.spin_slider = UIHorizontalSlider(
            pygame.Rect(x2+80, y, 150, 22),
            start_value=self.physics.species_config.wave_phase_speed[0],
            value_range=(-5.0, 5.0),
            manager=self.manager,
            container=self.species_window
        )
        y += 30
        
        # Row 3: Spin dynamics
        UILabel(pygame.Rect(x, y, 80, 20), "Spin Rate:", 
                manager=self.manager, container=self.species_window)
        self.base_spin_slider = UIHorizontalSlider(
            pygame.Rect(x+80, y, 150, 22),
            start_value=self.physics.species_config.base_spin_rate[0],
            value_range=(-3.0, 3.0),
            manager=self.manager,
            container=self.species_window
        )
        
        UILabel(pygame.Rect(x2, y, 80, 20), "Spin Fric:", 
                manager=self.manager, container=self.species_window)
        self.spin_friction_slider = UIHorizontalSlider(
            pygame.Rect(x2+80, y, 150, 22),
            start_value=self.physics.species_config.spin_friction[0],
            value_range=(0.1, 5.0),
            manager=self.manager,
            container=self.species_window
        )

    def _setup_performance_panel(self):
        """Performance monitoring panel."""
        rect = pygame.Rect(self.cfg.width - 200, self.cfg.height - 100, 190, 90)
        self.perf_panel = UIPanel(
            relative_rect=rect,
            starting_height=1,
            manager=self.manager
        )
        
        y = 5
        self.perf_fps_label = UILabel(pygame.Rect(10, y, 170, 20), "Avg FPS: --",
                                       manager=self.manager, container=self.perf_panel)
        y += 22
        self.perf_physics_label = UILabel(pygame.Rect(10, y, 170, 20), "Physics: -- ms",
                                          manager=self.manager, container=self.perf_panel)
        y += 22
        self.perf_render_label = UILabel(pygame.Rect(10, y, 170, 20), "Render: -- ms",
                                         manager=self.manager, container=self.perf_panel)

    def _add_section_header(self, y: int, text: str):
        """Add a section header to main panel."""
        UILabel(pygame.Rect(10, y, 250, 22), f"=== {text} ===", 
                manager=self.manager, container=self.hud_panel)
    
    def _add_section_header_to(self, y: int, text: str, container):
        """Add a section header to specified container."""
        UILabel(pygame.Rect(10, y, 250, 22), f"=== {text} ===", 
                manager=self.manager, container=container)

    def _refresh_matrix_buttons(self):
        """Update matrix button values from physics engine."""
        mat = self.physics.rules[self.active_rule_idx].matrix
        for (r, c), btn in self.matrix_buttons.items():
            if r < mat.shape[0] and c < mat.shape[1]:
                val = mat[r, c]
                btn.set_text(f"{val:.1f}")
        
        current_rule = self.physics.rules[self.active_rule_idx]
        self.radius_slider.set_current_value(current_rule.max_radius)
        self.strength_slider.set_current_value(current_rule.strength)

    def _update_rule_list(self):
        """Update the rule list display."""
        new_list = [f"{'[x]' if r.enabled else '[ ]'} {r.name}" for r in self.physics.rules]
        self.rule_list_box.set_item_list(new_list)

    def update(self, dt):
        """Called each frame to sync GUI with physics."""
        # Update config from sliders
        self.cfg.friction = self.friction_slider.get_current_value()
        self.cfg.dt = self.dt_slider.get_current_value()
        self.cfg.target_temperature = self.temp_slider.get_current_value()
        self.cfg.thermostat_coupling = self.coupling_slider.get_current_value()
        self.cfg.max_velocity = self.maxvel_slider.get_current_value()
        self.cfg.particle_scale = self.size_slider.get_current_value()
        
        # Update value labels
        self.friction_value.set_text(f"{self.cfg.friction:.2f}")
        self.dt_value.set_text(f"{self.cfg.dt:.4f}")
        self.temp_value.set_text(f"{self.cfg.target_temperature:.2f}")
        self.coupling_value.set_text(f"{self.cfg.thermostat_coupling:.2f}")
        self.maxvel_value.set_text(f"{self.cfg.max_velocity:.1f}")
        self.size_label.set_text(f"{self.cfg.particle_scale:.2f}x")
        self.worldsize_value.set_text(f"{self.cfg.world_size:.0f}")
        
        # Update active rule
        if self.physics.rules:
            current_rule = self.physics.rules[self.active_rule_idx]
            current_rule.max_radius = self.radius_slider.get_current_value()
            current_rule.strength = self.strength_slider.get_current_value()
            self.strength_value.set_text(f"{current_rule.strength:.1f}")
            self.radius_value.set_text(f"{current_rule.max_radius:.2f}")
        
        # Update species config
        idx = self.active_species_idx
        sc = self.physics.species_config
        if idx < len(sc.wave_freq):
            sc.radius[idx] = self.species_radius_slider.get_current_value()
            sc.wave_freq[idx] = round(self.freq_slider.get_current_value())
            sc.wave_amp[idx] = self.amp_slider.get_current_value()
            sc.wave_phase_speed[idx] = self.spin_slider.get_current_value()
            sc.base_spin_rate[idx] = self.base_spin_slider.get_current_value()
            sc.spin_friction[idx] = self.spin_friction_slider.get_current_value()
        
        # Update particle count label
        self.count_label.set_text(f"Particles: {self.physics.state.active}")
        
        # Compute energy
        vel = self.physics.state.vel[:self.physics.state.active]
        ke = 0.5 * np.sum(vel * vel)
        self.energy_label.set_text(f"Energy: {ke:.1f}")

    def update_performance(self, fps: float, physics_ms: float = 0, render_ms: float = 0):
        """Update performance display."""
        self.fps_label.set_text(f"FPS: {fps:.1f}")
        
        self.fps_history.append(fps)
        if len(self.fps_history) > 60:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        self.perf_fps_label.set_text(f"Avg FPS: {avg_fps:.1f}")
        self.perf_physics_label.set_text(f"Physics: {physics_ms:.1f} ms")
        self.perf_render_label.set_text(f"Render: {render_ms:.1f} ms")

    def _apply_preset(self, preset_name: str):
        """Apply a simulation preset."""
        if preset_name == "Small":
            cfg = SimulationConfig.small_world()
        elif preset_name == "Large":
            cfg = SimulationConfig.large_world()
        elif preset_name == "Huge":
            cfg = SimulationConfig.huge_world()
        elif preset_name == "Classic":
            cfg = SimulationConfig.classic_emergence()
            # Apply classic Haskell rules
            self.cfg.world_size = cfg.world_size
            self.cfg.num_particles = cfg.num_particles
            self.cfg.num_types = cfg.num_types
            self.cfg.dt = cfg.dt
            self.cfg.friction = cfg.friction
            self.cfg.slowdown_factor = cfg.slowdown_factor
            self.cfg.collision_damping = cfg.collision_damping
            self.cfg.thermostat_enabled = cfg.thermostat_enabled
            self.cfg.max_velocity = cfg.max_velocity
            self.cfg.wave_repulsion_strength = cfg.wave_repulsion_strength
            self.cfg.default_max_radius_frac = cfg.default_max_radius_frac
            self.cfg.default_min_radius_frac = cfg.default_min_radius_frac
            self.physics.set_species_count(cfg.num_types)
            self.physics.set_active_count(cfg.num_particles)
            self.physics.setup_classic_rules()
            self.physics.exclusion_enabled = False
            self._update_ui_from_config()
            self._rebuild_matrix_grid()
            self._update_rule_list()
            return
        elif preset_name == "Emergence":
            cfg = SimulationConfig.emergence_advanced()
            self.cfg.world_size = cfg.world_size
            self.cfg.num_particles = cfg.num_particles
            self.cfg.num_types = cfg.num_types
            self.cfg.dt = cfg.dt
            self.cfg.friction = cfg.friction
            self.cfg.slowdown_factor = cfg.slowdown_factor
            self.cfg.thermostat_enabled = cfg.thermostat_enabled
            self.cfg.max_velocity = cfg.max_velocity
            self.cfg.wave_repulsion_strength = cfg.wave_repulsion_strength
            self.physics.set_species_count(cfg.num_types)
            self.physics.set_active_count(cfg.num_particles)
            self.physics.exclusion_enabled = True
            self._update_ui_from_config()
            self._rebuild_matrix_grid()
            self._update_rule_list()
            return
        elif preset_name == "Dense":
            self.cfg.num_particles = 30000
            self.cfg.num_types = 8
            self.cfg.world_size = 50.0
            self.physics.set_species_count(8)
            self.physics.set_active_count(30000)
            return
        else:
            cfg = SimulationConfig.default()
        
        # Apply preset values
        self.cfg.world_size = cfg.world_size
        self.cfg.num_particles = cfg.num_particles
        self.cfg.num_types = cfg.num_types
        self.cfg.dt = cfg.dt
        self.cfg.friction = cfg.friction
        
        self.physics.set_species_count(cfg.num_types)
        self.physics.set_active_count(cfg.num_particles)
        
        # Update UI
        self._update_ui_from_config()
        self._rebuild_matrix_grid()
        self._update_rule_list()
    
    def _update_ui_from_config(self):
        """Update all UI elements from current config."""
        self.worldsize_slider.set_current_value(self.cfg.world_size)
        self.dt_slider.set_current_value(self.cfg.dt)
        self.friction_slider.set_current_value(self.cfg.friction)
        self.maxvel_slider.set_current_value(self.cfg.max_velocity)
        self.input_particles.set_text(str(self.cfg.num_particles))
        self.input_species.set_text(str(self.cfg.num_types))

    def _save_config(self, filepath="pyparticles_config.json"):
        """Save current configuration to JSON."""
        data = {
            "simulation": {
                "num_particles": self.cfg.num_particles,
                "num_types": self.cfg.num_types,
                "world_size": self.cfg.world_size,
                "friction": self.cfg.friction,
                "dt": self.cfg.dt,
                "particle_scale": self.cfg.particle_scale,
                "render_mode": self.cfg.render_mode.value,
                "target_temp": self.cfg.target_temperature,
                "coupling": self.cfg.thermostat_coupling,
                "max_velocity": self.cfg.max_velocity,
            },
            "rules": [
                {
                    "name": r.name,
                    "matrix": r.matrix.tolist(),
                    "max_r": r.max_radius,
                    "min_r": r.min_radius,
                    "strength": r.strength,
                    "force_type": int(r.force_type),
                    "enabled": r.enabled
                } for r in self.physics.rules
            ],
            "species": {
                "radius": self.physics.species_config.radius.tolist(),
                "wave_freq": self.physics.species_config.wave_freq.tolist(),
                "wave_amp": self.physics.species_config.wave_amp.tolist(),
                "wave_phase_speed": self.physics.species_config.wave_phase_speed.tolist(),
                "spin_inertia": self.physics.species_config.spin_inertia.tolist(),
                "spin_friction": self.physics.species_config.spin_friction.tolist(),
                "base_spin_rate": self.physics.species_config.base_spin_rate.tolist(),
            }
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[GUI] Saved config to {filepath}")

    def _load_config(self, filepath):
        """Load configuration from JSON."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            sim = data.get("simulation", {})
            self.cfg.num_particles = sim.get("num_particles", self.cfg.num_particles)
            self.cfg.world_size = sim.get("world_size", self.cfg.world_size)
            num_types = sim.get("num_types", self.cfg.num_types)
            self.cfg.friction = sim.get("friction", self.cfg.friction)
            self.cfg.dt = sim.get("dt", self.cfg.dt)
            self.cfg.particle_scale = sim.get("particle_scale", 1.0)
            self.cfg.target_temperature = sim.get("target_temp", self.cfg.target_temperature)
            self.cfg.thermostat_coupling = sim.get("coupling", self.cfg.thermostat_coupling)
            self.cfg.max_velocity = sim.get("max_velocity", self.cfg.max_velocity)
            
            self.physics.set_active_count(self.cfg.num_particles)
            self.physics.set_species_count(num_types)
            
            rule_data = data.get("rules", [])
            self.physics.rules = []
            for r in rule_data:
                rule = InteractionRule(
                    name=r["name"],
                    force_type=ForceType(r["force_type"]),
                    matrix=np.array(r["matrix"], dtype=np.float32),
                    max_radius=r["max_r"],
                    min_radius=r["min_r"],
                    strength=r.get("strength", 1.0),
                    enabled=r.get("enabled", True)
                )
                self.physics.rules.append(rule)
                
            sp_data = data.get("species", {})
            sc = self.physics.species_config
            if "radius" in sp_data: sc.radius = np.array(sp_data["radius"], dtype=np.float32)
            if "wave_freq" in sp_data: sc.wave_freq = np.array(sp_data["wave_freq"], dtype=np.float32)
            if "wave_amp" in sp_data: sc.wave_amp = np.array(sp_data["wave_amp"], dtype=np.float32)
            if "wave_phase_speed" in sp_data: sc.wave_phase_speed = np.array(sp_data["wave_phase_speed"], dtype=np.float32)
            if "spin_inertia" in sp_data: sc.spin_inertia = np.array(sp_data["spin_inertia"], dtype=np.float32)
            if "spin_friction" in sp_data: sc.spin_friction = np.array(sp_data["spin_friction"], dtype=np.float32)
            if "base_spin_rate" in sp_data: sc.base_spin_rate = np.array(sp_data["base_spin_rate"], dtype=np.float32)
            
            # Update UI sliders
            self.friction_slider.set_current_value(self.cfg.friction)
            self.dt_slider.set_current_value(self.cfg.dt)
            self.temp_slider.set_current_value(self.cfg.target_temperature)
            self.coupling_slider.set_current_value(self.cfg.thermostat_coupling)
            self.maxvel_slider.set_current_value(self.cfg.max_velocity)
            self.size_slider.set_current_value(self.cfg.particle_scale)
            self.worldsize_slider.set_current_value(self.cfg.world_size)
            
            self.input_particles.set_text(str(self.cfg.num_particles))
            self.input_species.set_text(str(self.cfg.num_types))
            
            self._update_rule_list()
            self._rebuild_matrix_grid()
            print(f"[GUI] Loaded config from {filepath}")
            
        except Exception as e:
            print(f"[GUI] Error loading config: {e}")

    def handle_event(self, event):
        """Process pygame_gui events."""
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.pause_btn:
                self.paused = not self.paused
                self.pause_btn.set_text("Resume" if self.paused else "Pause")
            elif event.ui_element == self.reset_btn:
                self.physics.reset()
            elif event.ui_element == self.btn_set_particles:
                try:
                    n = int(self.input_particles.get_text())
                    self.physics.set_active_count(n)
                    self.cfg.num_particles = n
                except ValueError:
                    pass
            elif event.ui_element == self.btn_set_species:
                try:
                    n = int(self.input_species.get_text())
                    self.physics.set_species_count(n)
                    self.cfg.num_types = n
                    self._rebuild_matrix_grid()
                    self._update_rule_list()
                except ValueError:
                    pass
            elif event.ui_element == self.save_btn:
                self._save_config()
            elif event.ui_element == self.load_btn:
                self.file_dialog = UIFileDialog(
                    rect=pygame.Rect(100, 100, 400, 400),
                    manager=self.manager,
                    window_title="Load Config",
                    initial_file_path=".",
                    allow_picking_directories=False
                )
            elif event.ui_element == self.randomize_species_btn:
                idx = self.active_species_idx
                sc = self.physics.species_config
                if idx < len(sc.wave_freq):
                    sc.wave_freq[idx] = np.random.randint(2, 8)
                    sc.wave_amp[idx] = np.random.uniform(0.01, 0.1)
                    sc.wave_phase_speed[idx] = np.random.uniform(-3.0, 3.0)
                    sc.base_spin_rate[idx] = np.random.uniform(-2.0, 2.0)
                    self._refresh_species_sliders()
            elif event.ui_element == self.randomize_all_btn:
                sc = self.physics.species_config
                n = self.cfg.num_types
                sc.wave_freq = np.random.randint(2, 8, n).astype(np.float32)
                sc.wave_amp = np.random.uniform(0.01, 0.1, n).astype(np.float32)
                sc.wave_phase_speed = np.random.uniform(-3.0, 3.0, n).astype(np.float32)
                sc.base_spin_rate = np.random.uniform(-2.0, 2.0, n).astype(np.float32)
                self._refresh_species_sliders()

            # Matrix button clicks
            for (r, c), btn in self.matrix_buttons.items():
                if event.ui_element == btn:
                    mat = self.physics.rules[self.active_rule_idx].matrix
                    val = mat[r, c]
                    val -= 0.5
                    if val < -1.0: val = 1.0
                    mat[r, c] = val
                    btn.set_text(f"{val:.1f}")

        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.rule_selector:
                for i, r in enumerate(self.physics.rules):
                    if r.name == event.text:
                        self.active_rule_idx = i
                        self._refresh_matrix_buttons()
                        break
            elif event.ui_element == self.species_selector:
                # Extract type number from "Type X"
                idx = int(event.text.split()[-1])
                self.active_species_idx = idx
                self._refresh_species_sliders()
            elif event.ui_element == self.preset_selector:
                self._apply_preset(event.text)
            elif event.ui_element == self.render_mode_selector:
                # Will be handled by renderer
                pass

        elif event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION:
            if event.ui_element == self.rule_list_box:
                text = event.text
                name = text[4:]  # Strip "[x] " or "[ ] "
                for r in self.physics.rules:
                    if r.name == name:
                        r.enabled = not r.enabled
                        break
                self._update_rule_list()
        
        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.worldsize_slider:
                # World size change requires physics reinit
                new_size = self.worldsize_slider.get_current_value()
                if abs(new_size - self.cfg.world_size) > 1.0:
                    self.cfg.world_size = new_size
                    # Re-scale species config
                    self.physics.species_config = type(self.physics.species_config).default(
                        self.cfg.num_types, new_size
                    )
                
        if event.type == pygame_gui.UI_WINDOW_CLOSE:
            if event.ui_element == self.file_dialog:
                self.file_dialog = None
                
        if event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
            if event.ui_element == self.file_dialog:
                path = event.text
                self._load_config(path)

    def _refresh_species_sliders(self):
        """Refresh species sliders with current species data."""
        idx = self.active_species_idx
        sc = self.physics.species_config
        if idx < len(sc.wave_freq):
            self.species_radius_slider.set_current_value(sc.radius[idx])
            self.freq_slider.set_current_value(sc.wave_freq[idx])
            self.amp_slider.set_current_value(sc.wave_amp[idx])
            self.spin_slider.set_current_value(sc.wave_phase_speed[idx])
            self.base_spin_slider.set_current_value(sc.base_spin_rate[idx])
            self.spin_friction_slider.set_current_value(sc.spin_friction[idx])

    def get_render_mode(self) -> int:
        """Get current render mode as int for shader."""
        mode_map = {"Standard": 0, "Wave": 1, "Energy": 2, "Minimal": 3}
        return mode_map.get(self.render_mode_selector.selected_option[0], 0)

