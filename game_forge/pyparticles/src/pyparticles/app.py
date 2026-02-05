"""
Application Loop.
Unified OpenGL Rendering.
"""
import sys
import os
import argparse
import pygame
import pygame_gui
import time
import numpy as np
import json

if __name__ == "__main__": # pragma: no cover
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from .core.types import SimulationConfig, RenderMode
from .physics.engine import PhysicsEngine
from .rendering.gl_renderer import GLCanvas
from .ui.gui import SimulationGUI 

def main():
    parser = argparse.ArgumentParser(description="Eidosian PyParticles V5 (OpenGL Only)")
    parser.add_argument("--num", "-n", type=int, default=5000)
    parser.add_argument("--types", "-t", type=int, default=6)
    parser.add_argument("--jit-warmup", action="store_true", default=True)
    
    args = parser.parse_args()
    
    cfg = SimulationConfig.default()
    cfg.num_particles = args.num
    cfg.num_types = args.types
    cfg.render_mode = RenderMode.OPENGL
    
    # Force minimal physics settings for stability
    cfg.dt = 0.005 # Lower default DT
    
    print("[System] Initializing Physics...")
    physics = PhysicsEngine(cfg)
    
    if args.jit_warmup:
        print("[System] JIT Warmup...")
        t0 = time.time()
        physics.update(0.001)
        print(f"[System] JIT Ready in {time.time()-t0:.2f}s")
        
    print("[System] Initializing OpenGL Renderer...")
    
    # Pygame Init
    pygame.init()
    pygame.display.set_caption("Eidosian PyParticles V5 (Ultra)")
    
    # Request OpenGL 3.3+ Core
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    
    screen = pygame.display.set_mode((cfg.width, cfg.height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    
    try:
        canvas = GLCanvas(cfg)
    except Exception as e:
        print(f"[Fatal] OpenGL Init Failed: {e}")
        return

    # UI Setup
    ui_manager = pygame_gui.UIManager((cfg.width, cfg.height))
    ui_surface = pygame.Surface((cfg.width, cfg.height), pygame.SRCALPHA)
    
    clock = pygame.time.Clock()
    running = True
    paused = False
    
    gui = SimulationGUI(ui_manager, cfg, physics)
    
    print("[System] Starting Loop.")
    
    while running:
        dt_ms = clock.tick(60)
        # Cap max dt to avoid physics explosion during lag spikes
        dt_sim = min(dt_ms / 1000.0, 0.05) 
        
        fps = clock.get_fps()
        gui.fps_label.set_text(f"FPS: {fps:.1f}")
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.VIDEORESIZE:
                canvas.resize(event.w, event.h)
                ui_manager.set_window_resolution((event.w, event.h))
                ui_surface = pygame.Surface((event.w, event.h), pygame.SRCALPHA)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    gui.pause_btn.set_text("Resume" if paused else "Pause")
                    gui.paused = paused
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
            ui_manager.process_events(event)
            gui.handle_event(event)
            
        ui_manager.update(dt_sim)
        gui.update(dt_sim)
        
        if not paused:
            # Sub-stepping for stability?
            # If physics is fast enough, run 2 steps of 0.5*dt
            # For now, single step.
            physics.update(cfg.dt) # Use fixed physics timestep
            
        # Render
        canvas.render(
            physics.state.pos, 
            physics.state.vel,
            physics.state.colors,
            physics.state.angle,
            physics._pack_species(),
            physics.state.active,
            fps
        )
        
        # UI Overlay
        ui_surface.fill((0,0,0,0))
        ui_manager.draw_ui(ui_surface)
        canvas.render_ui(ui_surface)
        
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__": # pragma: no cover
    main()