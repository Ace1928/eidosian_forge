"""
Application Loop.
"""
import sys
import os
import argparse
import pygame
import pygame_gui
import time
import numpy as np
import json

# Adjust path if run directly
if __name__ == "__main__": # pragma: no cover
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from .core.types import SimulationConfig, RenderMode
from .physics.engine import PhysicsEngine
from .rendering.canvas import Canvas
from .ui.gui import SimulationGUI 

def main():
    parser = argparse.ArgumentParser(description="Eidosian PyParticles V2")
    parser.add_argument("--num", "-n", type=int, default=5000)
    parser.add_argument("--types", "-t", type=int, default=6)
    parser.add_argument("--mode", type=str, default="sprites", choices=["sprites", "pixels", "glow"])
    parser.add_argument("--jit-warmup", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Config
    cfg = SimulationConfig.default()
    cfg.num_particles = args.num
    cfg.num_types = args.types
    cfg.render_mode = RenderMode(args.mode)
    
    # Systems
    print("[System] Initializing Physics...")
    physics = PhysicsEngine(cfg)
    
    if args.jit_warmup:
        print("[System] JIT Warmup...")
        t0 = time.time()
        physics.update(0.01)
        print(f"[System] JIT Ready in {time.time()-t0:.2f}s")
        
    print("[System] Initializing Renderer...")
    canvas = Canvas(cfg)
    
    # UI Theme
    ui_manager = pygame_gui.UIManager((cfg.width, cfg.height)) 
    
    # Logic Loop
    clock = pygame.time.Clock()
    running = True
    paused = False
    
    # Connect GUI
    gui = SimulationGUI(ui_manager, cfg, physics)
    
    print("[System] Starting Loop.")
    
    while running:
        dt_ms = clock.tick(60)
        dt = dt_ms / 1000.0
        fps = clock.get_fps()
        
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False # pragma: no cover
            
            if event.type == pygame.VIDEORESIZE:
                # Handle Resize
                canvas.resize(event.w, event.h)
                ui_manager.set_window_resolution((event.w, event.h))
                # Update physics bounds? Physics assumes [-1, 1]. Renderer handles mapping.
                # So physics doesn't care about pixels. Good.
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    gui.pause_btn.set_text("Resume" if paused else "Pause")
                    gui.paused = paused
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
            ui_manager.process_events(event)
            gui.handle_event(event)
            
        # Update UI
        ui_manager.update(dt)
        gui.update(dt)
        
        # Update Physics
        if not paused:
            # GUI slider updates cfg.dt, so we use that
            physics.update() 
            
        # Render
        canvas.render(
            physics.state.pos, 
            physics.state.colors, 
            physics.state.active,
            fps
        )
        
        ui_manager.draw_ui(canvas.screen)
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__": # pragma: no cover
    main()
