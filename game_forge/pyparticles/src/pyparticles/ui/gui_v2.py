"""
Eidosian PyParticles V6.2 - Modern Collapsible GUI

Clean, modular, high-performance UI with collapsible sections.
Designed for maximum configurability and aesthetic appeal.
"""
import json

import numpy as np
import pygame
import pygame_gui
from pygame_gui.elements import (
    UIHorizontalSlider,
    UILabel,
    UIButton,
    UIDropDownMenu,
    UIPanel,
)
from pygame_gui.windows import UIFileDialog
from ..core.types import SimulationConfig, SpeciesConfig


class CollapsibleSection:
    """A collapsible UI section with header and content."""
    
    def __init__(self, manager, container, title: str, y_start: int, width: int = 260):
        self.manager = manager
        self.container = container
        self.title = title
        self.width = width
        self.collapsed = False
        self.elements = []
        self.element_offsets = []
        self.y_start = y_start
        self.content_height = 0
        
        # Header button (acts as toggle)
        self.header = UIButton(
            pygame.Rect(5, y_start, width, 24),
            f"v {title}",
            manager=manager,
            container=container,
        )
        self.content_y = y_start + 26

    def _register_element(self, element, x: int, y_offset: int) -> None:
        """Track element offsets for section relayout."""
        self.elements.append(element)
        self.element_offsets.append((element, x, y_offset))
        
    def toggle(self):
        """Toggle collapsed state."""
        self.collapsed = not self.collapsed
        symbol = ">" if self.collapsed else "v"
        self.header.set_text(f"{symbol} {self.title}")
        for el in self.elements:
            el.visible = not self.collapsed
        
    def add_slider(self, label: str, value: float, range_: tuple, y_offset: int) -> tuple:
        """Add a labeled slider. Returns (label_element, slider_element, value_label)."""
        y = self.content_y + y_offset
        
        label_rect = pygame.Rect(10, y, 120, 20)  # Wider labels
        lbl = UILabel(
            label_rect,
            label,
            manager=self.manager,
            container=self.container,
        )
        
        value_rect = pygame.Rect(185, y, 70, 20)  # Wider value labels
        val_lbl = UILabel(
            value_rect,
            f"{value:.2f}",
            manager=self.manager,
            container=self.container,
        )
        
        slider_rect = pygame.Rect(10, y + 20, self.width - 20, 20)
        slider = UIHorizontalSlider(
            slider_rect,
            start_value=max(range_[0], min(range_[1], value)),
            value_range=range_,
            manager=self.manager,
            container=self.container
        )
        
        y_offset = y - self.content_y
        self._register_element(lbl, label_rect.x, y_offset)
        self._register_element(val_lbl, value_rect.x, y_offset)
        self._register_element(slider, slider_rect.x, y_offset + 20)
        self.content_height = max(self.content_height, y_offset + 44)
        return lbl, slider, val_lbl
    
    def add_button(
        self, text: str, y_offset: int, width: int = None, x_offset: int = 10
    ) -> UIButton:
        """Add a button."""
        y = self.content_y + y_offset
        w = width or (self.width - 20)
        btn = UIButton(
            pygame.Rect(x_offset, y, w, 26),
            text,
            manager=self.manager,
            container=self.container
        )
        self._register_element(btn, x_offset, y_offset)
        self.content_height = max(self.content_height, y_offset + 30)
        return btn
    
    def add_dropdown(self, options: list, default: str, y_offset: int) -> UIDropDownMenu:
        """Add a dropdown menu."""
        y = self.content_y + y_offset
        rect = pygame.Rect(10, y, self.width - 20, 26)
        dd = UIDropDownMenu(
            options_list=options,
            starting_option=default,
            relative_rect=rect,
            manager=self.manager,
            container=self.container
        )
        self._register_element(dd, rect.x, y_offset)
        self.content_height = max(self.content_height, y_offset + 30)
        return dd
    
    def add_label(self, text: str, y_offset: int) -> UILabel:
        """Add a label."""
        y = self.content_y + y_offset
        rect = pygame.Rect(10, y, self.width - 20, 20)
        lbl = UILabel(
            rect,
            text,
            manager=self.manager,
            container=self.container
        )
        self._register_element(lbl, rect.x, y_offset)
        self.content_height = max(self.content_height, y_offset + 22)
        return lbl
    
    def get_total_height(self) -> int:
        """Get total height including header."""
        if self.collapsed:
            return 26
        return 26 + self.content_height + 5


class SimulationGUI:
    """
    Modern collapsible GUI for PyParticles V6.2.
    
    Features:
    - Collapsible sections for clean layout
    - Real-time parameter adjustment
    - Proper world size scaling
    - Force toggle system
    - Preset management
    - Performance monitoring
    """
    
    def __init__(self, manager: pygame_gui.UIManager, config: SimulationConfig, physics_engine):
        self.manager = manager
        self.cfg = config
        self.physics = physics_engine
        self.window_size = (config.width, config.height)
        
        self.paused = False
        self.file_dialog = None
        
        # Performance tracking
        self.fps_history = []
        
        # Create main panel
        panel_width = 280
        panel_height = config.height - 20
        self.main_panel = UIPanel(
            pygame.Rect(10, 10, panel_width, panel_height),
            starting_height=1,
            manager=manager
        )
        
        self._build_ui()
    
    def _build_ui(self):
        """Build all UI sections."""
        y = 5
        
        # === STATUS SECTION ===
        self.status_section = CollapsibleSection(
            self.manager, self.main_panel, "STATUS", y
        )
        self.fps_label = self.status_section.add_label("FPS: --", 0)
        self.particle_label = self.status_section.add_label(f"Particles: {self.cfg.num_particles}", 22)
        self.energy_label = self.status_section.add_label("Energy: --", 44)
        y += self.status_section.get_total_height()
        
        # === SIMULATION CONTROLS ===
        self.sim_section = CollapsibleSection(
            self.manager, self.main_panel, "SIMULATION", y
        )
        self.pause_btn = self.sim_section.add_button("Pause", 0, 120)
        self.reset_btn = self.sim_section.add_button("Reset", 0, x_offset=135)
        
        # Particle count
        _, self.particle_slider, self.particle_val = self.sim_section.add_slider(
            "Particles", self.cfg.num_particles, (100, 50000), 30
        )
        
        # Species count
        _, self.species_slider, self.species_val = self.sim_section.add_slider(
            "Species", self.cfg.num_types, (2, 32), 75
        )
        
        y += self.sim_section.get_total_height()
        
        # === WORLD SETTINGS ===
        self.world_section = CollapsibleSection(
            self.manager, self.main_panel, "WORLD", y
        )
        _, self.worldsize_slider, self.worldsize_val = self.world_section.add_slider(
            "World Size", self.cfg.world_size, (2.0, 500.0), 0
        )
        _, self.size_slider, self.size_val = self.world_section.add_slider(
            "Particle Scale", self.cfg.particle_scale, (0.1, 5.0), 45
        )
        y += self.world_section.get_total_height()
        
        # === PHYSICS ===
        self.physics_section = CollapsibleSection(
            self.manager, self.main_panel, "PHYSICS", y
        )
        _, self.dt_slider, self.dt_val = self.physics_section.add_slider(
            "Time Step", self.cfg.dt, (0.001, 0.02), 0
        )
        _, self.friction_slider, self.friction_val = self.physics_section.add_slider(
            "Friction", self.cfg.friction, (0.0, 2.0), 45
        )
        _, self.slowdown_slider, self.slowdown_val = self.physics_section.add_slider(
            "Slowdown", self.cfg.slowdown_factor, (0.1, 1.0), 90
        )
        _, self.maxvel_slider, self.maxvel_val = self.physics_section.add_slider(
            "Max Velocity", self.cfg.max_velocity, (1.0, 50.0), 135
        )
        y += self.physics_section.get_total_height()
        
        # === THERMOSTAT ===
        self.thermo_section = CollapsibleSection(
            self.manager, self.main_panel, "THERMOSTAT", y
        )
        self.thermo_toggle = self.thermo_section.add_button(
            "Enabled" if self.cfg.thermostat_enabled else "Disabled", 0, 120
        )
        _, self.temp_slider, self.temp_val = self.thermo_section.add_slider(
            "Temperature", self.cfg.target_temperature, (0.0, 5.0), 30
        )
        _, self.coupling_slider, self.coupling_val = self.thermo_section.add_slider(
            "Coupling", self.cfg.thermostat_coupling, (0.0, 0.5), 75
        )
        y += self.thermo_section.get_total_height()
        
        # === ADVANCED PHYSICS ===
        self.advanced_section = CollapsibleSection(
            self.manager, self.main_panel, "ADVANCED PHYSICS", y
        )
        self.wave_toggle = self.advanced_section.add_button(
            f"Wave: {'ON' if self.cfg.wave_repulsion_strength > 0 else 'OFF'}", 0, 80
        )
        self.exclusion_toggle = self.advanced_section.add_button(
            f"Exclusion: {'ON' if self.physics.exclusion_enabled else 'OFF'}", 0, 100, x_offset=95
        )
        self.spin_toggle = self.advanced_section.add_button(
            f"Spin: {'ON' if self.physics.spin_flip_enabled else 'OFF'}", 0, 70, x_offset=200
        )
        
        _, self.wave_strength_slider, self.wave_strength_val = self.advanced_section.add_slider(
            "Wave Strength", self.cfg.wave_repulsion_strength, (0.0, 20.0), 30
        )
        _, self.exclusion_strength_slider, self.exclusion_val = self.advanced_section.add_slider(
            "Exclusion Str", self.physics.exclusion_strength, (0.0, 30.0), 75
        )
        _, self.spin_coupling_slider, self.spin_coupling_val = self.advanced_section.add_slider(
            "Spin Coupling", self.physics.spin_coupling_strength, (0.0, 2.0), 120
        )
        y += self.advanced_section.get_total_height()
        
        # === FORCE RULES ===
        self.forces_section = CollapsibleSection(
            self.manager, self.main_panel, "FORCE RULES", y
        )
        rule_names = [r.name[:20] for r in self.physics.rules] or ["None"]
        self.rule_dropdown = self.forces_section.add_dropdown(rule_names, rule_names[0], 0)
        
        self.rule_toggle_btn = self.forces_section.add_button("Toggle Rule", 30, 120)
        
        _, self.rule_strength_slider, self.rule_strength_val = self.forces_section.add_slider(
            "Strength", self.physics.rules[0].strength if self.physics.rules else 1.0, (0.0, 20.0), 60
        )
        _, self.rule_radius_slider, self.rule_radius_val = self.forces_section.add_slider(
            "Radius", self.physics.rules[0].max_radius if self.physics.rules else 1.0, (0.01, 100.0), 105
        )
        y += self.forces_section.get_total_height()
        
        # === PRESETS ===
        self.preset_section = CollapsibleSection(
            self.manager, self.main_panel, "PRESETS", y
        )
        presets = ["Default", "Classic", "Emergence", "Small", "Large", "Huge"]
        self.preset_dropdown = self.preset_section.add_dropdown(presets, "Default", 0)
        
        self.save_btn = self.preset_section.add_button("Save", 30, 120)
        self.load_btn = self.preset_section.add_button("Load", 30, 120, x_offset=135)
        y += self.preset_section.get_total_height()
        
        # === RENDER MODE ===
        self.render_section = CollapsibleSection(
            self.manager, self.main_panel, "RENDER", y
        )
        modes = ["Standard", "Wave", "Energy", "Minimal"]
        self.render_dropdown = self.render_section.add_dropdown(modes, "Standard", 0)
        
        # Store all sections for collapse management
        self.sections = [
            self.status_section, self.sim_section, self.world_section,
            self.physics_section, self.thermo_section, self.advanced_section,
            self.forces_section, self.preset_section, self.render_section
        ]
        
        # Active rule tracking
        self.active_rule_idx = 0
    
    def handle_event(self, event):
        """Handle pygame_gui events."""
        
        # Section header clicks (collapse/expand)
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            for section in self.sections:
                if event.ui_element == section.header:
                    section.toggle()
                    self._relayout_sections()
                    return
            
            # Pause/Reset buttons
            if event.ui_element == self.pause_btn:
                self.paused = not self.paused
                self.pause_btn.set_text("Resume" if self.paused else "Pause")
            
            elif event.ui_element == self.reset_btn:
                self.physics.reset()
            
            # Thermostat toggle
            elif event.ui_element == self.thermo_toggle:
                self.cfg.thermostat_enabled = not self.cfg.thermostat_enabled
                self.thermo_toggle.set_text("Enabled" if self.cfg.thermostat_enabled else "Disabled")
            
            # Wave toggle
            elif event.ui_element == self.wave_toggle:
                if self.cfg.wave_repulsion_strength > 0:
                    self._saved_wave_strength = self.cfg.wave_repulsion_strength
                    self.cfg.wave_repulsion_strength = 0.0
                else:
                    self.cfg.wave_repulsion_strength = getattr(self, '_saved_wave_strength', 5.0)
                self.wave_toggle.set_text(f"Wave: {'ON' if self.cfg.wave_repulsion_strength > 0 else 'OFF'}")
                self.wave_strength_slider.set_current_value(self.cfg.wave_repulsion_strength)
            
            # Exclusion toggle
            elif event.ui_element == self.exclusion_toggle:
                self.physics.exclusion_enabled = not self.physics.exclusion_enabled
                self.exclusion_toggle.set_text(f"Exclusion: {'ON' if self.physics.exclusion_enabled else 'OFF'}")
            
            # Spin toggle
            elif event.ui_element == self.spin_toggle:
                self.physics.spin_flip_enabled = not self.physics.spin_flip_enabled
                self.spin_toggle.set_text(f"Spin: {'ON' if self.physics.spin_flip_enabled else 'OFF'}")
            
            # Rule toggle
            elif event.ui_element == self.rule_toggle_btn:
                if self.physics.rules:
                    rule = self.physics.rules[self.active_rule_idx]
                    rule.enabled = not rule.enabled
            
            # Save/Load
            elif event.ui_element == self.save_btn:
                self._save_config()
            elif event.ui_element == self.load_btn:
                self._show_load_dialog()
        
        # Dropdown changes
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.preset_dropdown:
                self._apply_preset(event.text)
            
            elif event.ui_element == self.rule_dropdown:
                for i, r in enumerate(self.physics.rules):
                    if r.name[:20] == event.text:
                        self.active_rule_idx = i
                        self._refresh_rule_sliders()
                        break
        
        # Slider changes (for world size which needs immediate action)
        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.worldsize_slider:
                self._handle_world_size_change()
            elif event.ui_element == self.particle_slider:
                new_count = int(self.particle_slider.get_current_value())
                if new_count != self.physics.state.active:
                    self.physics.set_active_count(new_count)
            elif event.ui_element == self.species_slider:
                new_types = int(self.species_slider.get_current_value())
                if new_types != self.cfg.num_types:
                    self.physics.set_species_count(new_types)
                    self._refresh_rule_dropdown()
        
        # File dialog
        if event.type == pygame_gui.UI_WINDOW_CLOSE:
            if event.ui_element == self.file_dialog:
                self.file_dialog = None
        
        if event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
            if event.ui_element == self.file_dialog:
                self._load_config(event.text)
    
    def _relayout_sections(self):
        """Relayout sections after collapse/expand."""
        y = 5
        for section in self.sections:
            section.header.set_relative_position((5, y))
            section.y_start = y
            section.content_y = y + 26
            
            # Reposition all elements in section
            for el, x, y_offset in section.element_offsets:
                el.set_relative_position((x, section.content_y + y_offset))
            
            y += section.get_total_height()
    
    def _handle_world_size_change(self):
        """Handle world size slider change with proper physics update."""
        new_size = self.worldsize_slider.get_current_value()
        old_size = self.cfg.world_size
        
        if abs(new_size - old_size) < 0.5:
            return
        
        # Scale factor for existing particles
        scale = new_size / old_size
        
        # Update config
        self.cfg.world_size = new_size
        
        # Scale particle positions
        n = self.physics.state.active
        self.physics.state.pos[:n] *= scale
        
        # Clamp to new bounds
        half = new_size / 2.0
        self.physics.state.pos[:n] = np.clip(self.physics.state.pos[:n], -half, half)
        
        # Update species config radii
        self.physics.species_config = SpeciesConfig.default(self.cfg.num_types, new_size)
        
        # Update rules radii
        for rule in self.physics.rules:
            rule.max_radius *= scale
            rule.min_radius *= scale
        
        # Rebuild spatial grid
        self.physics.max_interaction_radius = self.physics._get_max_radius()
        self.physics.cell_size = max(self.physics.max_interaction_radius, 0.1 * new_size)
        self.physics.grid_w = int(new_size / self.physics.cell_size) + 2
        self.physics.grid_h = int(new_size / self.physics.cell_size) + 2
        self.physics.grid_counts = np.zeros((self.physics.grid_h, self.physics.grid_w), dtype=np.int32)
        self.physics.grid_cells = np.zeros(
            (self.physics.grid_h, self.physics.grid_w, self.physics.max_per_cell), dtype=np.int32
        )
        
        # Invalidate force cache
        self.physics._invalidate_cache()
        
        # Update rule radius slider range
        self.rule_radius_slider.value_range = (0.01, new_size)
    
    def _refresh_rule_sliders(self):
        """Refresh rule sliders for active rule."""
        if not self.physics.rules:
            return
        rule = self.physics.rules[self.active_rule_idx]
        self.rule_strength_slider.set_current_value(rule.strength)
        self.rule_radius_slider.set_current_value(min(rule.max_radius, self.cfg.world_size))
    
    def _refresh_rule_dropdown(self):
        """Refresh rule dropdown after species change."""
        rule_names = [r.name[:20] for r in self.physics.rules] or ["None"]
        # Rebuild dropdown (pygame_gui doesn't support dynamic option updates easily)
        # For now just update active_rule_idx
        self.active_rule_idx = 0
    
    def _apply_preset(self, preset_name: str):
        """Apply a simulation preset with proper physics update."""
        if preset_name == "Classic":
            cfg = SimulationConfig.classic_emergence()
            self.physics.set_species_count(cfg.num_types)
            self.physics.set_active_count(cfg.num_particles)
            self.physics.setup_classic_rules()
            self.physics.exclusion_enabled = False
            self.physics.spin_flip_enabled = False
        elif preset_name == "Emergence":
            cfg = SimulationConfig.emergence_advanced()
            self.physics.set_species_count(cfg.num_types)
            self.physics.set_active_count(cfg.num_particles)
            self.physics.exclusion_enabled = True
            self.physics.spin_flip_enabled = True
        elif preset_name == "Small":
            cfg = SimulationConfig.small_world()
        elif preset_name == "Large":
            cfg = SimulationConfig.large_world()
        elif preset_name == "Huge":
            cfg = SimulationConfig.huge_world()
        else:
            cfg = SimulationConfig.default()
        
        # Apply config values
        self.cfg.world_size = cfg.world_size
        self.cfg.num_particles = cfg.num_particles
        self.cfg.num_types = cfg.num_types
        self.cfg.dt = cfg.dt
        self.cfg.friction = cfg.friction
        self.cfg.slowdown_factor = cfg.slowdown_factor
        self.cfg.thermostat_enabled = cfg.thermostat_enabled
        self.cfg.target_temperature = cfg.target_temperature
        self.cfg.max_velocity = cfg.max_velocity
        self.cfg.wave_repulsion_strength = cfg.wave_repulsion_strength
        
        # Update physics
        if preset_name not in ["Classic", "Emergence"]:
            self.physics.set_species_count(cfg.num_types)
            self.physics.set_active_count(cfg.num_particles)
        
        # Rebuild grid for new world size
        self._handle_world_size_change_force(cfg.world_size)
        
        # Update all sliders
        self._sync_sliders_from_config()
    
    def _handle_world_size_change_force(self, new_size: float):
        """Force world size change without scale factor (for presets)."""
        self.cfg.world_size = new_size
        self.physics.species_config = SpeciesConfig.default(self.cfg.num_types, new_size)
        
        # Reinit particles in new world
        self.physics.reset()
        
        # Rebuild grid
        self.physics.max_interaction_radius = self.physics._get_max_radius()
        self.physics.cell_size = max(self.physics.max_interaction_radius, 0.1 * new_size)
        self.physics.grid_w = int(new_size / self.physics.cell_size) + 2
        self.physics.grid_h = int(new_size / self.physics.cell_size) + 2
        self.physics.grid_counts = np.zeros((self.physics.grid_h, self.physics.grid_w), dtype=np.int32)
        self.physics.grid_cells = np.zeros(
            (self.physics.grid_h, self.physics.grid_w, self.physics.max_per_cell), dtype=np.int32
        )
        self.physics._invalidate_cache()
    
    def _sync_sliders_from_config(self):
        """Sync all slider values from current config."""
        self.worldsize_slider.set_current_value(self.cfg.world_size)
        self.particle_slider.set_current_value(self.cfg.num_particles)
        self.species_slider.set_current_value(self.cfg.num_types)
        self.dt_slider.set_current_value(self.cfg.dt)
        self.friction_slider.set_current_value(self.cfg.friction)
        self.slowdown_slider.set_current_value(self.cfg.slowdown_factor)
        self.maxvel_slider.set_current_value(self.cfg.max_velocity)
        self.temp_slider.set_current_value(self.cfg.target_temperature)
        self.coupling_slider.set_current_value(self.cfg.thermostat_coupling)
        self.size_slider.set_current_value(self.cfg.particle_scale)
        self.wave_strength_slider.set_current_value(self.cfg.wave_repulsion_strength)
        self.exclusion_strength_slider.set_current_value(self.physics.exclusion_strength)
        self.spin_coupling_slider.set_current_value(self.physics.spin_coupling_strength)
        
        # Update toggles
        self.thermo_toggle.set_text("Enabled" if self.cfg.thermostat_enabled else "Disabled")
        self.wave_toggle.set_text(f"Wave: {'ON' if self.cfg.wave_repulsion_strength > 0 else 'OFF'}")
        self.exclusion_toggle.set_text(f"Exclusion: {'ON' if self.physics.exclusion_enabled else 'OFF'}")
        self.spin_toggle.set_text(f"Spin: {'ON' if self.physics.spin_flip_enabled else 'OFF'}")
    
    def _save_config(self, filepath: str = "pyparticles_config.json"):
        """Save current configuration to JSON."""
        data = {
            "simulation": {
                "num_particles": self.cfg.num_particles,
                "num_types": self.cfg.num_types,
                "world_size": self.cfg.world_size,
                "friction": self.cfg.friction,
                "slowdown_factor": self.cfg.slowdown_factor,
                "dt": self.cfg.dt,
                "particle_scale": self.cfg.particle_scale,
                "target_temperature": self.cfg.target_temperature,
                "thermostat_coupling": self.cfg.thermostat_coupling,
                "thermostat_enabled": self.cfg.thermostat_enabled,
                "max_velocity": self.cfg.max_velocity,
                "wave_repulsion_strength": self.cfg.wave_repulsion_strength,
            },
            "physics": {
                "exclusion_enabled": self.physics.exclusion_enabled,
                "exclusion_strength": self.physics.exclusion_strength,
                "spin_flip_enabled": self.physics.spin_flip_enabled,
                "spin_coupling_strength": self.physics.spin_coupling_strength,
                "spin_enabled": self.physics.spin_enabled,
            },
            "rules": [
                {
                    "name": r.name,
                    "matrix": r.matrix.tolist(),
                    "max_radius": r.max_radius,
                    "min_radius": r.min_radius,
                    "strength": r.strength,
                    "force_type": int(r.force_type),
                    "enabled": r.enabled
                } for r in self.physics.rules
            ]
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[Config] Saved to {filepath}")
        except Exception as e:
            print(f"[Error] Failed to save: {e}")
    
    def _show_load_dialog(self):
        """Show file dialog for loading config."""
        self.file_dialog = UIFileDialog(
            pygame.Rect(100, 100, 400, 300),
            self.manager,
            "Load Configuration",
            initial_file_path=".",
            allow_picking_directories=False
        )
    
    def _load_config(self, filepath: str):
        """Load configuration from JSON."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            sim = data.get("simulation", {})
            self.cfg.world_size = sim.get("world_size", self.cfg.world_size)
            self.cfg.num_particles = sim.get("num_particles", self.cfg.num_particles)
            self.cfg.num_types = sim.get("num_types", self.cfg.num_types)
            self.cfg.friction = sim.get("friction", self.cfg.friction)
            self.cfg.slowdown_factor = sim.get("slowdown_factor", self.cfg.slowdown_factor)
            self.cfg.dt = sim.get("dt", self.cfg.dt)
            self.cfg.particle_scale = sim.get("particle_scale", self.cfg.particle_scale)
            self.cfg.target_temperature = sim.get("target_temperature", self.cfg.target_temperature)
            self.cfg.thermostat_coupling = sim.get("thermostat_coupling", self.cfg.thermostat_coupling)
            self.cfg.thermostat_enabled = sim.get("thermostat_enabled", self.cfg.thermostat_enabled)
            self.cfg.max_velocity = sim.get("max_velocity", self.cfg.max_velocity)
            self.cfg.wave_repulsion_strength = sim.get("wave_repulsion_strength", self.cfg.wave_repulsion_strength)
            
            phys = data.get("physics", {})
            self.physics.exclusion_enabled = phys.get("exclusion_enabled", self.physics.exclusion_enabled)
            self.physics.exclusion_strength = phys.get("exclusion_strength", self.physics.exclusion_strength)
            self.physics.spin_flip_enabled = phys.get("spin_flip_enabled", self.physics.spin_flip_enabled)
            self.physics.spin_coupling_strength = phys.get("spin_coupling_strength", self.physics.spin_coupling_strength)
            self.physics.spin_enabled = phys.get("spin_enabled", self.physics.spin_enabled)
            
            # Apply changes
            self.physics.set_species_count(self.cfg.num_types)
            self.physics.set_active_count(self.cfg.num_particles)
            self._handle_world_size_change_force(self.cfg.world_size)
            self._sync_sliders_from_config()
            
            print(f"[Config] Loaded from {filepath}")
        except Exception as e:
            print(f"[Error] Failed to load: {e}")
    
    def update(self, dt):
        """Update GUI state each frame."""
        # Sync config from sliders
        self.cfg.dt = self.dt_slider.get_current_value()
        self.cfg.friction = self.friction_slider.get_current_value()
        self.cfg.slowdown_factor = self.slowdown_slider.get_current_value()
        self.cfg.max_velocity = self.maxvel_slider.get_current_value()
        self.cfg.target_temperature = self.temp_slider.get_current_value()
        self.cfg.thermostat_coupling = self.coupling_slider.get_current_value()
        self.cfg.particle_scale = self.size_slider.get_current_value()
        self.cfg.wave_repulsion_strength = self.wave_strength_slider.get_current_value()
        self.physics.exclusion_strength = self.exclusion_strength_slider.get_current_value()
        self.physics.spin_coupling_strength = self.spin_coupling_slider.get_current_value()
        
        # Update active rule
        if self.physics.rules:
            rule = self.physics.rules[self.active_rule_idx]
            rule.strength = self.rule_strength_slider.get_current_value()
            rule.max_radius = self.rule_radius_slider.get_current_value()
        
        # Update value labels
        self.dt_val.set_text(f"{self.cfg.dt:.4f}")
        self.friction_val.set_text(f"{self.cfg.friction:.2f}")
        self.slowdown_val.set_text(f"{self.cfg.slowdown_factor:.2f}")
        self.maxvel_val.set_text(f"{self.cfg.max_velocity:.1f}")
        self.temp_val.set_text(f"{self.cfg.target_temperature:.2f}")
        self.coupling_val.set_text(f"{self.cfg.thermostat_coupling:.2f}")
        self.size_val.set_text(f"{self.cfg.particle_scale:.2f}x")
        self.worldsize_val.set_text(f"{self.cfg.world_size:.0f}")
        self.wave_strength_val.set_text(f"{self.cfg.wave_repulsion_strength:.1f}")
        self.exclusion_val.set_text(f"{self.physics.exclusion_strength:.1f}")
        self.spin_coupling_val.set_text(f"{self.physics.spin_coupling_strength:.2f}")
        self.particle_val.set_text(f"{int(self.particle_slider.get_current_value())}")
        self.species_val.set_text(f"{int(self.species_slider.get_current_value())}")
        
        if self.physics.rules:
            rule = self.physics.rules[self.active_rule_idx]
            self.rule_strength_val.set_text(f"{rule.strength:.1f}")
            self.rule_radius_val.set_text(f"{rule.max_radius:.2f}")
        
        # Update status
        self.particle_label.set_text(f"Particles: {self.physics.state.active}")
        vel = self.physics.state.vel[:self.physics.state.active]
        ke = 0.5 * np.sum(vel * vel)
        self.energy_label.set_text(f"Energy: {ke:.1f}")
    
    def update_performance(self, fps: float, physics_ms: float = 0, render_ms: float = 0):
        """Update FPS display."""
        self.fps_label.set_text(f"FPS: {fps:.0f} | Phys: {physics_ms:.1f}ms | Rnd: {render_ms:.1f}ms")
    
    def get_render_mode(self) -> int:
        """Get current render mode as int for shader."""
        mode_map = {"Standard": 0, "Wave": 1, "Energy": 2, "Minimal": 3}
        selected = self.render_dropdown.selected_option
        if isinstance(selected, tuple):
            selected = selected[0]
        return mode_map.get(selected, 0)
