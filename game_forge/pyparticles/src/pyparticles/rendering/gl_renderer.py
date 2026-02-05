"""
ModernGL Renderer.
Uses instanced rendering for massive particle counts.
"""
import moderngl
import numpy as np
import pygame
import os
from typing import Optional
from ..core.types import SimulationConfig

class GLCanvas:
    def __init__(self, config: SimulationConfig):
        self.ctx = moderngl.create_context()
        self.cfg = config
        
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        
        self._init_particle_pipeline()
        self._init_ui_pipeline()
        
        # Coordinate State
        self.width = config.width
        self.height = config.height
        self.scale = self.width / 2.0
        self.offset_x = self.width / 2.0
        self.offset_y = self.height / 2.0
        
    def _init_particle_pipeline(self):
        base = os.path.dirname(__file__)
        with open(os.path.join(base, "shaders/particle.vert"), 'r') as f: vs = f.read()
        with open(os.path.join(base, "shaders/particle.geom"), 'r') as f: gs = f.read()
        with open(os.path.join(base, "shaders/particle.frag"), 'r') as f: fs = f.read()
        self.prog = self.ctx.program(vertex_shader=vs, geometry_shader=gs, fragment_shader=fs)
        
        # Buffer: pos(2), color(3), radius(1), freq(1), amp(1), angle(1)
        # Note: Energy removed from VBO for now as we removed it from State.
        # Wait, user said "remove biology" but "keep energy" for conservation/thermostat.
        # But ParticleState energy was "Metabolic Energy".
        # Kinetic Energy is derived (0.5 v^2).
        # Should we visualize KE?
        # Yes! High KE = Bright.
        # We need to calculate KE per particle and upload it?
        # Or just pass velocity magnitude?
        # Let's pass Velocity Magnitude as "Energy" for visualization.
        
        self.buffer_size = self.cfg.max_particles * 40
        self.vbo = self.ctx.buffer(reserve=self.buffer_size, dynamic=True)
        
        content = [
            (self.vbo, '2f 3f 1f 1f 1f 1f 1f', 'in_pos', 'in_color', 'in_radius', 'in_freq', 'in_amp', 'in_angle', 'in_energy')
        ]
        self.vao = self.ctx.vertex_array(self.prog, content)

    def _init_ui_pipeline(self):
        base = os.path.dirname(__file__)
        with open(os.path.join(base, "shaders/ui.vert"), 'r') as f: vs = f.read()
        with open(os.path.join(base, "shaders/ui.frag"), 'r') as f: fs = f.read()
        self.ui_prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        
        quad = np.array([
            -1.0, -1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 1.0,
            -1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
        ], dtype='f4')
        
        self.ui_vbo = self.ctx.buffer(quad.tobytes())
        self.ui_vao = self.ctx.vertex_array(self.ui_prog, [(self.ui_vbo, '2f 2f', 'in_vert', 'in_uv')])
        self.ui_tex = None

    def resize(self, w, h):
        self.width = w
        self.height = h
        self.ctx.viewport = (0, 0, w, h)
        self.scale = w / 2.0
        self.offset_x = w / 2.0
        self.offset_y = h / 2.0

    def render(self, pos, vel, colors, angle, species_params, active_count, fps):
        self.ctx.clear(0.05, 0.05, 0.07)
        
        if active_count > 0:
            from ..utils.colors import generate_hsv_palette
            if not hasattr(self, 'palette'):
                pal_list = generate_hsv_palette(self.cfg.num_types) 
                pal_int = np.array(pal_list, dtype=np.float32)
                self.palette = pal_int / 255.0
            
            p_colors = self.palette[colors[:active_count]]
            p_params = species_params[colors[:active_count]]
            
            # Calculate Speed for Energy Visualization
            # speed = sqrt(vx^2 + vy^2)
            # Just approximation for visual: |v|
            vel_slice = vel[:active_count]
            speeds = np.linalg.norm(vel_slice, axis=1).astype(np.float32)
            
            # Normalize speed for glow? 
            # Target Temp is ~0.5. So v ~ 1.0.
            # Multiply by some factor.
            
            data = np.empty((active_count, 10), dtype=np.float32)
            data[:, 0:2] = pos[:active_count]
            data[:, 2:5] = p_colors
            data[:, 5:8] = p_params[:, 0:3] # rad, freq, amp
            data[:, 8] = angle[:active_count]
            data[:, 9] = speeds * 2.0 # Scale factor for glow
            
            self.vbo.write(data.tobytes())
            
            self.prog['window_size'].value = (float(self.width), float(self.height))
            self.prog['scale'].value = self.scale
            self.prog['offset'].value = (self.offset_x, self.offset_y)
            
            self.vao.render(moderngl.POINTS, vertices=active_count)

    def render_ui(self, surface: pygame.Surface):
        data = surface.get_view('1')
        if self.ui_tex is None or self.ui_tex.size != surface.get_size():
            if self.ui_tex: self.ui_tex.release()
            self.ui_tex = self.ctx.texture(surface.get_size(), 4, data=data)
        else:
            self.ui_tex.write(data)
            
        self.ui_tex.use(0)
        self.ui_prog['ui_texture'].value = 0
        self.ctx.enable(moderngl.BLEND)
        self.ui_vao.render(moderngl.TRIANGLE_STRIP)