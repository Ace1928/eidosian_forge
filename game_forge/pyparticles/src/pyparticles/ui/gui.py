"""
Modern GUI Layer using pygame_gui.
Provides an extensible control panel for the simulation.
"""
import pygame
import pygame_gui
import json
from pygame_gui.elements import UIWindow, UIHorizontalSlider, UILabel, UIButton, UIPanel, UIDropDownMenu
from ..core.types import SimulationConfig
from ..utils.colors import generate_hsv_palette

class SimulationGUI:
    def __init__(self, manager: pygame_gui.UIManager, config: SimulationConfig, physics_engine):
        self.manager = manager
        self.cfg = config
        self.physics = physics_engine
        self.window_size = (config.width, config.height)
        
        # State
        self.paused = False
        self.active_rule_idx = 0
        
        self._setup_hud()
        self._setup_matrix_editor()

    def _setup_hud(self):
        """Create the main control sidebar."""
        rect = pygame.Rect(10, 10, 250, 500)
        self.hud_panel = UIPanel(
            relative_rect=rect,
            starting_height=1,
            manager=self.manager
        )
        
        y = 10
        UILabel(pygame.Rect(10, y, 230, 30), "EIDOSIAN CONTROLS", 
                manager=self.manager, container=self.hud_panel)
        y += 40
        
        self.fps_label = UILabel(pygame.Rect(10, y, 230, 20), "FPS: --", 
                                 manager=self.manager, container=self.hud_panel)
        y += 30
        
        self.count_label = UILabel(pygame.Rect(10, y, 230, 20), f"N: {self.cfg.num_particles}",
                                   manager=self.manager, container=self.hud_panel)
        y += 30
        
        # Friction
        UILabel(pygame.Rect(10, y, 230, 20), "Friction", manager=self.manager, container=self.hud_panel)
        y += 20
        self.friction_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 230, 25),
            start_value=self.cfg.friction,
            value_range=(0.0, 1.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 35
        
        # DT
        UILabel(pygame.Rect(10, y, 230, 20), "Time Step", manager=self.manager, container=self.hud_panel)
        y += 20
        self.dt_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 230, 25),
            start_value=self.cfg.dt,
            value_range=(0.001, 0.1),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 35
        
        # Rule Selector
        UILabel(pygame.Rect(10, y, 230, 20), "Active Force Rule", manager=self.manager, container=self.hud_panel)
        y += 20
        rule_names = [r.name for r in self.physics.rules]
        self.rule_selector = UIDropDownMenu(
            options_list=rule_names,
            starting_option=rule_names[0],
            relative_rect=pygame.Rect(10, y, 230, 30),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 40

        # Radius (For active rule)
        self.lbl_radius = UILabel(pygame.Rect(10, y, 230, 20), "Rule Max Radius", manager=self.manager, container=self.hud_panel)
        y += 20
        self.radius_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 230, 25),
            start_value=self.physics.rules[0].max_radius,
            value_range=(0.01, 1.0),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 35

        # Buttons
        self.pause_btn = UIButton(pygame.Rect(10, y, 110, 30), "Pause", 
                                  manager=self.manager, container=self.hud_panel)
        self.reset_btn = UIButton(pygame.Rect(130, y, 110, 30), "Reset",
                                  manager=self.manager, container=self.hud_panel)

    def _setup_matrix_editor(self):
        """Create a visual grid for editing interaction rules."""
        cell_size = 30
        grid_size = self.cfg.num_types * cell_size + 80 
        
        rect = pygame.Rect(self.cfg.width - grid_size - 10, 10, grid_size, grid_size)
        
        self.matrix_window = UIWindow(
            rect=rect,
            manager=self.manager,
            window_display_title="Interaction Matrix",
            resizable=True
        )
        
        self.matrix_buttons = {} # (row, col) -> button
        
        for r in range(self.cfg.num_types):
            for c in range(self.cfg.num_types):
                # Init with rule 0
                val = self.physics.rules[0].matrix[r, c]
                
                btn_rect = pygame.Rect(20 + c*cell_size, 20 + r*cell_size, cell_size-2, cell_size-2)
                
                btn = UIButton(
                    relative_rect=btn_rect,
                    text=f"{val:.1f}",
                    manager=self.manager,
                    container=self.matrix_window,
                    object_id=f"#matrix_btn_{r}_{c}" 
                )
                self.matrix_buttons[(r, c)] = btn
        
        save_btn_rect = pygame.Rect(20, self.cfg.num_types * cell_size + 30, 80, 30)
        self.save_btn = UIButton(
            relative_rect=save_btn_rect,
            text="Save",
            manager=self.manager,
            container=self.matrix_window
        )

    def _refresh_matrix_buttons(self):
        """Update button text to match active rule."""
        mat = self.physics.rules[self.active_rule_idx].matrix
        for (r, c), btn in self.matrix_buttons.items():
            val = mat[r, c]
            btn.set_text(f"{val:.1f}")
            
        # Also update radius slider
        self.radius_slider.set_current_value(self.physics.rules[self.active_rule_idx].max_radius)

    def update(self, dt):
        """Update GUI state based on sliders."""
        self.cfg.friction = self.friction_slider.get_current_value()
        self.cfg.dt = self.dt_slider.get_current_value()
        
        # Update active rule radius
        current_rule = self.physics.rules[self.active_rule_idx]
        current_rule.max_radius = self.radius_slider.get_current_value()

    def handle_event(self, event):
        """Handle UI-specific events."""
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.pause_btn:
                self.paused = not self.paused
                self.pause_btn.set_text("Resume" if self.paused else "Pause")
            elif event.ui_element == self.reset_btn:
                self.physics.reset()
            elif event.ui_element == self.save_btn:
                # Save All Rules
                data = {
                    "rules": [
                        {
                            "name": r.name,
                            "matrix": r.matrix.tolist(),
                            "max_r": r.max_radius,
                            "type": int(r.force_type)
                        } for r in self.physics.rules
                    ]
                }
                with open("rules_config.json", "w") as f:
                    json.dump(data, f)
                print("Rules saved to rules_config.json")
            
            # Matrix Buttons
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
                # Find index
                for i, r in enumerate(self.physics.rules):
                    if r.name == event.text:
                        self.active_rule_idx = i
                        self._refresh_matrix_buttons()
                        break
