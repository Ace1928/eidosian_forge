"""
Modern GUI Layer using pygame_gui.
Provides an extensible control panel for the simulation.
"""
import pygame
import pygame_gui
import json
from pygame_gui.elements import UIWindow, UIHorizontalSlider, UILabel, UIButton, UIPanel
from pygame_gui.windows import UIFileDialog
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
        
        self._setup_hud()
        self._setup_matrix_editor()

    def _setup_hud(self):
        """Create the main control sidebar."""
        rect = pygame.Rect(10, 10, 250, 450)
        self.hud_panel = UIPanel(
            relative_rect=rect,
            starting_layer_height=1,
            manager=self.manager
        )
        
        # Title
        y = 10
        UILabel(pygame.Rect(10, y, 230, 30), "EIDOSIAN CONTROLS", 
                manager=self.manager, container=self.hud_panel)
        y += 40
        
        # FPS Label
        self.fps_label = UILabel(pygame.Rect(10, y, 230, 20), "FPS: --", 
                                 manager=self.manager, container=self.hud_panel)
        y += 30
        
        # Particle Count
        self.count_label = UILabel(pygame.Rect(10, y, 230, 20), f"N: {self.cfg.num_particles}",
                                   manager=self.manager, container=self.hud_panel)
        y += 30
        
        # Controls
        # Friction
        UILabel(pygame.Rect(10, y, 230, 20), "Friction (Damping)", manager=self.manager, container=self.hud_panel)
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
        UILabel(pygame.Rect(10, y, 230, 20), "Time Step (DT)", manager=self.manager, container=self.hud_panel)
        y += 20
        self.dt_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 230, 25),
            start_value=self.cfg.dt,
            value_range=(0.001, 0.1),
            manager=self.manager,
            container=self.hud_panel
        )
        y += 35
        
        # Repulsion Radius
        UILabel(pygame.Rect(10, y, 230, 20), "Repulsion Radius", manager=self.manager, container=self.hud_panel)
        y += 20
        self.rep_slider = UIHorizontalSlider(
            pygame.Rect(10, y, 230, 25),
            start_value=self.cfg.max_radius * 0.3, # Approx default
            value_range=(0.01, self.cfg.max_radius),
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
        grid_size = self.cfg.num_types * cell_size + 80 # + padding and save button
        
        rect = pygame.Rect(self.cfg.width - grid_size - 10, 10, grid_size, grid_size)
        
        self.matrix_window = UIWindow(
            rect=rect,
            manager=self.manager,
            window_display_title="Interaction Matrix",
            resizable=True
        )
        
        self.matrix_buttons = {} # (row, col) -> button
        
        # We don't need colors for buttons themselves usually, but let's keep logic
        
        for r in range(self.cfg.num_types):
            for c in range(self.cfg.num_types):
                val = self.physics.matrix[r, c]
                
                btn_rect = pygame.Rect(20 + c*cell_size, 20 + r*cell_size, cell_size-2, cell_size-2)
                
                btn = UIButton(
                    relative_rect=btn_rect,
                    text=f"{val:.1f}",
                    manager=self.manager,
                    container=self.matrix_window,
                    object_id=f"#matrix_btn_{r}_{c}" 
                )
                self.matrix_buttons[(r, c)] = btn
        
        # Save Button in Matrix Window
        save_btn_rect = pygame.Rect(20, self.cfg.num_types * cell_size + 30, 80, 30)
        self.save_btn = UIButton(
            relative_rect=save_btn_rect,
            text="Save",
            manager=self.manager,
            container=self.matrix_window
        )

    def update(self, dt):
        """Update GUI state based on sliders."""
        # FPS passed in via set_text separately if needed, or we read from clock in main
        # But we need to update sliders to config
        self.cfg.friction = self.friction_slider.get_current_value()
        self.cfg.dt = self.dt_slider.get_current_value()

    def handle_event(self, event):
        """Handle UI-specific events."""
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.pause_btn:
                self.paused = not self.paused
                self.pause_btn.set_text("Resume" if self.paused else "Pause")
            elif event.ui_element == self.reset_btn:
                self.physics.reset()
            elif event.ui_element == self.save_btn:
                # Save Matrix
                data = {
                    "matrix": self.physics.matrix.tolist()
                }
                with open("matrix_config.json", "w") as f:
                    json.dump(data, f)
                print("Matrix saved to matrix_config.json")
            
            # Check matrix buttons
            for (r, c), btn in self.matrix_buttons.items():
                if event.ui_element == btn:
                    val = self.physics.matrix[r, c]
                    val -= 0.5
                    if val < -1.0: val = 1.0
                    self.physics.matrix[r, c] = val
                    btn.set_text(f"{val:.1f}")
