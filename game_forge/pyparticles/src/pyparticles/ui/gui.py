"""
Modern GUI Layer.
Updated for Procedural Physics (Thermostat, No Biology).
"""
import pygame
import pygame_gui
import json
import numpy as np
from pygame_gui.elements import (
    UIWindow, UIHorizontalSlider, UILabel, UIButton, UIPanel, 
    UIDropDownMenu, UITextEntryLine, UISelectionList
)
from pygame_gui.windows import UIFileDialog
from ..core.types import SimulationConfig, InteractionRule, ForceType, RenderMode

class SimulationGUI:
    def __init__(self, manager: pygame_gui.UIManager, config: SimulationConfig, physics_engine):
        self.manager = manager
        self.cfg = config
        self.physics = physics_engine
        self.window_size = (config.width, config.height)
        
        self.paused = False
        self.active_rule_idx = 0
        self.active_species_idx = 0
        self.file_dialog = None
        
        self._setup_hud()
        self._setup_matrix_editor()
        self._setup_species_editor()

    def _setup_hud(self):
        rect = pygame.Rect(10, 10, 260, 700)
        self.hud_panel = UIPanel(
            relative_rect=rect,
            starting_height=1,
            manager=self.manager
        )
        
        y = 10
        UILabel(pygame.Rect(10, y, 240, 30), "EIDOSIAN CONTROLS V4", 
                manager=self.manager, container=self.hud_panel)
        y += 35
        
        self.fps_label = UILabel(pygame.Rect(10, y, 240, 20), "FPS: --", 
                                 manager=self.manager, container=self.hud_panel)
        y += 20
        self.count_label = UILabel(pygame.Rect(10, y, 240, 20), f"Particles: {self.cfg.num_particles}",
                                   manager=self.manager, container=self.hud_panel)
        y += 30
        
        # Particles Config
        UILabel(pygame.Rect(10, y, 240, 20), "Total Particles:", manager=self.manager, container=self.hud_panel)
        y += 20
        self.input_particles = UITextEntryLine(
            pygame.Rect(10, y, 140, 30),
            manager=self.manager,
            container=self.hud_panel
        )
        self.input_particles.set_text(str(self.cfg.num_particles))
        self.btn_set_particles = UIButton(
            pygame.Rect(160, y, 80, 30), "Set",
            manager=self.manager, container=self.hud_panel
        )
        y += 40
        
        # Species Config
        UILabel(pygame.Rect(10, y, 240, 20), "Species Count:", manager=self.manager, container=self.hud_panel)
        y += 20
        self.input_species = UITextEntryLine(
            pygame.Rect(10, y, 140, 30),
            manager=self.manager,
            container=self.hud_panel
        )
        self.input_species.set_text(str(self.cfg.num_types))
        self.btn_set_species = UIButton(
            pygame.Rect(160, y, 80, 30), "Reset",
            manager=self.manager, container=self.hud_panel
        )
        y += 40

        # Sliders
        UILabel(pygame.Rect(10, y, 240, 20), "Friction (1.0 = None)", manager=self.manager, container=self.hud_panel)
        y += 20
        self.friction_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 240, 20),
            start_value=self.cfg.friction,
            value_range=(0.0, 1.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 30
        
        UILabel(pygame.Rect(10, y, 240, 20), "Time Step", manager=self.manager, container=self.hud_panel)
        y += 20
        self.dt_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 240, 20),
            start_value=self.cfg.dt,
            value_range=(0.001, 0.05),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 30
        
        # Thermostat Sliders
        UILabel(pygame.Rect(10, y, 240, 20), "Target Temp (KE)", manager=self.manager, container=self.hud_panel)
        y += 20
        self.temp_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 240, 20),
            start_value=self.cfg.target_temperature,
            value_range=(0.0, 2.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 30
        
        UILabel(pygame.Rect(10, y, 240, 20), "Thermostat Coupling", manager=self.manager, container=self.hud_panel)
        y += 20
        self.coupling_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 240, 20),
            start_value=self.cfg.thermostat_coupling,
            value_range=(0.0, 1.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 40
        
        # Rules
        UILabel(pygame.Rect(10, y, 240, 20), "Active Force Layers", manager=self.manager, container=self.hud_panel)
        y += 25
        self.rule_list_box = UISelectionList(
            relative_rect=pygame.Rect(10, y, 240, 100),
            item_list=[],
            manager=self.manager,
            container=self.hud_panel,
            allow_multi_select=True
        )
        self._update_rule_list()
        y += 110

        # Matrix
        UILabel(pygame.Rect(10, y, 240, 20), "Edit Rule Matrix", manager=self.manager, container=self.hud_panel)
        y += 20
        rule_names = [r.name for r in self.physics.rules]
        self.rule_selector = UIDropDownMenu(
            options_list=rule_names,
            starting_option=rule_names[0],
            relative_rect=pygame.Rect(10, y, 240, 30),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 35
        
        self.lbl_radius = UILabel(pygame.Rect(10, y, 240, 20), "Rule Max Radius", manager=self.manager, container=self.hud_panel)
        y += 20
        self.radius_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 240, 20),
            start_value=self.physics.rules[0].max_radius,
            value_range=(0.01, 1.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 30
        
        self.lbl_strength = UILabel(pygame.Rect(10, y, 240, 20), "Rule Strength", manager=self.manager, container=self.hud_panel)
        y += 20
        self.strength_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 240, 20),
            start_value=self.physics.rules[0].strength,
            value_range=(0.0, 5.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 40

        self.pause_btn = UIButton(pygame.Rect(10, y, 115, 30), "Pause", 
                                  manager=self.manager, container=self.hud_panel)
        self.reset_btn = UIButton(pygame.Rect(135, y, 115, 30), "Reset Sim",
                                  manager=self.manager, container=self.hud_panel)

    def _setup_matrix_editor(self):
        cell_size = 30
        grid_size = min(300, self.cfg.num_types * cell_size) + 60
        rect = pygame.Rect(self.cfg.width - grid_size - 10, 10, grid_size, grid_size + 40)
        
        self.matrix_window = UIWindow(
            rect=rect,
            manager=self.manager,
            window_display_title="Matrix",
            resizable=True
        )
        
        y_off = self.cfg.num_types * cell_size + 30
        self.save_btn = UIButton(
            relative_rect=pygame.Rect(20, y_off, 70, 30),
            text="Save",
            manager=self.manager,
            container=self.matrix_window
        )
        self.load_btn = UIButton(
            relative_rect=pygame.Rect(100, y_off, 70, 30),
            text="Load",
            manager=self.manager,
            container=self.matrix_window
        )
        
        self.matrix_buttons = {} 
        self._rebuild_matrix_grid()
        
    def _rebuild_matrix_grid(self):
        for btn in self.matrix_buttons.values():
            btn.kill()
        self.matrix_buttons = {}
        cell_size = 30
        
        for r in range(self.cfg.num_types):
            for c in range(self.cfg.num_types):
                if r < self.physics.rules[0].matrix.shape[0] and c < self.physics.rules[0].matrix.shape[1]:
                    val = self.physics.rules[self.active_rule_idx].matrix[r, c]
                    btn_rect = pygame.Rect(10 + c*cell_size, 10 + r*cell_size, cell_size-2, cell_size-2)
                    btn = UIButton(
                        relative_rect=btn_rect,
                        text=f"{val:.1f}",
                        manager=self.manager,
                        container=self.matrix_window
                    )
                    self.matrix_buttons[(r, c)] = btn

    def _setup_species_editor(self):
        w = 500
        h = 120 # Reduced height (removed biology)
        rect = pygame.Rect((self.cfg.width - w)//2, self.cfg.height - h - 10, w, h)
        
        self.species_window = UIWindow(
            rect=rect,
            manager=self.manager,
            window_display_title="Species Wave Config",
            resizable=False
        )
        
        x = 10
        y = 10
        UILabel(pygame.Rect(x, y, 100, 30), "Select Type:", 
                manager=self.manager, container=self.species_window)
        
        type_options = [str(i) for i in range(self.cfg.num_types)]
        self.species_selector = UIDropDownMenu(
            options_list=type_options,
            starting_option="0",
            relative_rect=pygame.Rect(x+110, y, 80, 30),
            manager=self.manager,
            container=self.species_window
        )
        
        x = 10
        y += 40
        UILabel(pygame.Rect(x, y, 80, 20), "Freq (Lobes)", manager=self.manager, container=self.species_window)
        self.freq_slider = UIHorizontalSlider(
            pygame.Rect(x+90, y, 120, 20),
            start_value=self.physics.species_config.wave_freq[0],
            value_range=(1.0, 8.0),
            manager=self.manager,
            container=self.species_window
        )
        
        x += 230
        UILabel(pygame.Rect(x, y, 80, 20), "Amplitude", manager=self.manager, container=self.species_window)
        self.amp_slider = UIHorizontalSlider(
            pygame.Rect(x+90, y, 120, 20),
            start_value=self.physics.species_config.wave_amp[0],
            value_range=(0.0, 0.1),
            manager=self.manager,
            container=self.species_window
        )
        
        x = 10
        y += 30
        UILabel(pygame.Rect(x, y, 80, 20), "Spin", manager=self.manager, container=self.species_window)
        self.spin_slider = UIHorizontalSlider(
            pygame.Rect(x+90, y, 120, 20),
            start_value=self.physics.species_config.wave_phase_speed[0],
            value_range=(-5.0, 5.0),
            manager=self.manager,
            container=self.species_window
        )

    def _refresh_matrix_buttons(self):
        mat = self.physics.rules[self.active_rule_idx].matrix
        for (r, c), btn in self.matrix_buttons.items():
            if r < mat.shape[0] and c < mat.shape[1]:
                val = mat[r, c]
                btn.set_text(f"{val:.1f}")
        
        current_rule = self.physics.rules[self.active_rule_idx]
        self.radius_slider.set_current_value(current_rule.max_radius)
        self.strength_slider.set_current_value(current_rule.strength)

    def _update_rule_list(self):
        new_list = [f"{'[x]' if r.enabled else '[ ]'} {r.name}" for r in self.physics.rules]
        self.rule_list_box.set_item_list(new_list)

    def update(self, dt):
        self.cfg.friction = self.friction_slider.get_current_value()
        self.cfg.dt = self.dt_slider.get_current_value()
        self.cfg.target_temperature = self.temp_slider.get_current_value()
        self.cfg.thermostat_coupling = self.coupling_slider.get_current_value()
        
        current_rule = self.physics.rules[self.active_rule_idx]
        current_rule.max_radius = self.radius_slider.get_current_value()
        current_rule.strength = self.strength_slider.get_current_value()
        
        idx = self.active_species_idx
        sc = self.physics.species_config
        if idx < len(sc.wave_freq):
            sc.wave_freq[idx] = round(self.freq_slider.get_current_value())
            sc.wave_amp[idx] = self.amp_slider.get_current_value()
            sc.wave_phase_speed[idx] = self.spin_slider.get_current_value()

    def _save_config(self, filepath="pyparticles_config.json"):
        data = {
            "simulation": {
                "num_particles": self.cfg.num_particles,
                "num_types": self.cfg.num_types,
                "friction": self.cfg.friction,
                "dt": self.cfg.dt,
                "render_mode": self.cfg.render_mode.value,
                "target_temp": self.cfg.target_temperature,
                "coupling": self.cfg.thermostat_coupling
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
                "wave_phase_speed": self.physics.species_config.wave_phase_speed.tolist()
            }
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved config to {filepath}")

    def _load_config(self, filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            sim = data.get("simulation", {})
            self.cfg.num_particles = sim.get("num_particles", self.cfg.num_particles)
            num_types = sim.get("num_types", self.cfg.num_types)
            self.cfg.friction = sim.get("friction", self.cfg.friction)
            self.cfg.dt = sim.get("dt", self.cfg.dt)
            self.cfg.target_temperature = sim.get("target_temp", self.cfg.target_temperature)
            self.cfg.thermostat_coupling = sim.get("coupling", self.cfg.thermostat_coupling)
            
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
            
            self.friction_slider.set_current_value(self.cfg.friction)
            self.dt_slider.set_current_value(self.cfg.dt)
            self.temp_slider.set_current_value(self.cfg.target_temperature)
            self.coupling_slider.set_current_value(self.cfg.thermostat_coupling)
            
            self.input_particles.set_text(str(self.cfg.num_particles))
            self.input_species.set_text(str(self.cfg.num_types))
            
            self._update_rule_list()
            self._rebuild_matrix_grid()
            print(f"Loaded config from {filepath}")
            
        except Exception as e:
            print(f"Error loading config: {e}")

    def handle_event(self, event):
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
                except ValueError:
                    pass
            elif event.ui_element == self.btn_set_species:
                try:
                    n = int(self.input_species.get_text())
                    self.physics.set_species_count(n)
                    self._rebuild_matrix_grid()
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
                idx = int(event.text)
                self.active_species_idx = idx
                sc = self.physics.species_config
                if idx < len(sc.wave_freq):
                    self.freq_slider.set_current_value(sc.wave_freq[idx])
                    self.amp_slider.set_current_value(sc.wave_amp[idx])
                    self.spin_slider.set_current_value(sc.wave_phase_speed[idx])

        elif event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION:
            if event.ui_element == self.rule_list_box:
                text = event.text
                name = text[4:]
                for r in self.physics.rules:
                    if r.name == name:
                        r.enabled = not r.enabled
                        break
                self._update_rule_list()
                
        if event.type == pygame_gui.UI_WINDOW_CLOSE:
            if event.ui_element == self.file_dialog:
                self.file_dialog = None
                
        if event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
            if event.ui_element == self.file_dialog:
                path = event.text
                self._load_config(path)
