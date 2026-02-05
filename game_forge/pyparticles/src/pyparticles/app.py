"""
Eidosian PyParticles V6 - Application Loop
Advanced OpenGL Rendering with full GUI control.
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
    parser = argparse.ArgumentParser(description="Eidosian PyParticles V6 (Ultra)")
    parser.add_argument("--num", "-n", type=int, default=None, help="Number of particles")
    parser.add_argument("--types", "-t", type=int, default=None, help="Number of species")
    parser.add_argument("--world-size", "-w", type=float, default=None, help="World size")
    parser.add_argument("--jit-warmup", action="store_true", default=True)
    parser.add_argument("--preset", choices=['small', 'default', 'large', 'huge'], default='default')
    
    args = parser.parse_args()
    
    # Select preset
    if args.preset == 'small':
        cfg = SimulationConfig.small_world()
    elif args.preset == 'large':
        cfg = SimulationConfig.large_world()
    elif args.preset == 'huge':
        cfg = SimulationConfig.huge_world()
    else:
        cfg = SimulationConfig.default()
    
    # Override with args if provided
    if args.num is not None:
        cfg.num_particles = args.num
    if args.types is not None:
        cfg.num_types = args.types
    if args.world_size is not None:
        cfg.world_size = args.world_size
    cfg.render_mode = RenderMode.OPENGL
    
    # Validate and warn
    warnings = cfg.validate()
    for w in warnings:
        print(f"[Warning] {w}")
    
    print(f"[System] Config: {cfg.num_particles} particles, {cfg.num_types} types, world_size={cfg.world_size}")
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
    pygame.display.set_caption(f"Eidosian PyParticles V6 | {cfg.num_particles} particles | world={cfg.world_size}")
    
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
    print("[System] Controls: SPACE=pause, ESC=quit, mouse=interact")
    
    # Performance tracking
    physics_time = 0
    render_time = 0
    
    while running:
        dt_ms = clock.tick(60)
        # Cap max dt to avoid physics explosion during lag spikes
        dt_sim = min(dt_ms / 1000.0, 0.05) 
        
        fps = clock.get_fps()
        gui.update_performance(fps, physics_time * 1000, render_time * 1000)
        
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
                if event.key == pygame.K_r:
                    physics.reset()
                if event.key == pygame.K_1:
                    cfg.particle_scale = 0.5
                if event.key == pygame.K_2:
                    cfg.particle_scale = 1.0
                if event.key == pygame.K_3:
                    cfg.particle_scale = 2.0
                    
            ui_manager.process_events(event)
            gui.handle_event(event)
            
        ui_manager.update(dt_sim)
        gui.update(dt_sim)
        
        # Physics update with timing
        if not paused:
            t0 = time.perf_counter()
            physics.update(cfg.dt)  # Uses substeps internally
            physics_time = time.perf_counter() - t0
        
        # Render with timing
        t0 = time.perf_counter()
        canvas.render(
            physics.state.pos, 
            physics.state.vel,
            physics.state.colors,
            physics.state.angle,
            physics._pack_species(),
            physics.state.active,
            fps,
            cfg.world_size,
            cfg.particle_scale,
            gui.get_render_mode()
        )
        render_time = time.perf_counter() - t0
        
        # UI Overlay
        ui_surface.fill((0,0,0,0))
        ui_manager.draw_ui(ui_surface)
        canvas.render_ui(ui_surface)
        
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__": # pragma: no cover
    main()