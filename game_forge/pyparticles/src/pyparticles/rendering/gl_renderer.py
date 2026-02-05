"""
Eidosian PyParticles V6 - ModernGL Renderer

GPU-accelerated particle rendering with advanced wave visualization,
energy effects, multiple render modes, and view transforms.
"""

import moderngl
import numpy as np
import pygame
import os
import time as time_module
from typing import Optional
from ..core.types import SimulationConfig


class RenderMode:
    """Render mode constants matching shader."""
    STANDARD = 0
    WAVE_ONLY = 1
    ENERGY_ONLY = 2
    MINIMAL = 3


class GLCanvas:
    """
    OpenGL rendering canvas using ModernGL.
    
    Features:
    - Geometry shader billboard expansion with LOD
    - Wave-deformed particle shapes
    - Energy-based glow effects with thermal visualization
    - Multiple render modes for analysis
    - View transforms (pan, zoom)
    - UI overlay compositing
    """
    
    def __init__(self, config: SimulationConfig):
        self.ctx = moderngl.create_context()
        self.cfg = config
        
        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        
        self._init_particle_pipeline()
        self._init_ui_pipeline()
        
        # Window state
        self.width = config.width
        self.height = config.height
        
        # View transforms
        self.view_center = (0.0, 0.0)
        self.view_scale = 1.0
        
        # Render settings
        self.render_mode = RenderMode.STANDARD
        self.glow_intensity = 1.0
        self.edge_sharpness = 1.0
        self.particle_scale = 1.0
        self.ambient_color = (0.02, 0.02, 0.04)
        
        # Animation time
        self.start_time = time_module.time()
        
        # Color palette cache
        self.palette = None
        self.palette_size = 0
        
    def _init_particle_pipeline(self):
        """Initialize particle rendering shader pipeline."""
        base = os.path.dirname(__file__)
        
        with open(os.path.join(base, "shaders/particle.vert"), 'r') as f:
            vs = f.read()
        with open(os.path.join(base, "shaders/particle.geom"), 'r') as f:
            gs = f.read()
        with open(os.path.join(base, "shaders/particle.frag"), 'r') as f:
            fs = f.read()
        
        self.prog = self.ctx.program(
            vertex_shader=vs,
            geometry_shader=gs,
            fragment_shader=fs
        )
        
        # Vertex buffer: pos(2) + color(3) + radius(1) + freq(1) + amp(1) + angle(1) + energy(1) = 10 floats
        self.buffer_size = self.cfg.max_particles * 40  # 10 floats * 4 bytes
        self.vbo = self.ctx.buffer(reserve=self.buffer_size, dynamic=True)
        
        # Vertex array with attribute layout
        content = [
            (self.vbo, '2f 3f 1f 1f 1f 1f 1f', 
             'in_pos', 'in_color', 'in_radius', 'in_freq', 'in_amp', 'in_angle', 'in_energy')
        ]
        self.vao = self.ctx.vertex_array(self.prog, content)
        
        # Cache uniform locations for performance
        self._cache_uniforms()

    def _cache_uniforms(self):
        """Cache uniform setters for better performance."""
        self.uniforms = {}
        uniform_names = [
            'window_size', 'view_center', 'view_scale', 'time',
            'particle_scale', 'render_mode', 'glow_intensity',
            'edge_sharpness', 'ambient_color', 'wave_viz_mode'
        ]
        for name in uniform_names:
            if name in self.prog:
                self.uniforms[name] = self.prog[name]
    
    def _set_uniform(self, name: str, value):
        """Safely set uniform if it exists."""
        if name in self.uniforms:
            self.uniforms[name].value = value

    def _init_ui_pipeline(self):
        """Initialize UI overlay rendering pipeline."""
        base = os.path.dirname(__file__)
        
        with open(os.path.join(base, "shaders/ui.vert"), 'r') as f:
            vs = f.read()
        with open(os.path.join(base, "shaders/ui.frag"), 'r') as f:
            fs = f.read()
        
        self.ui_prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        
        # Fullscreen quad for UI overlay
        quad = np.array([
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 0.0,
        ], dtype='f4')
        
        self.ui_vbo = self.ctx.buffer(quad.tobytes())
        self.ui_vao = self.ctx.vertex_array(
            self.ui_prog, 
            [(self.ui_vbo, '2f 2f', 'in_vert', 'in_uv')]
        )
        self.ui_tex = None

    def resize(self, w: int, h: int):
        """Handle window resize."""
        self.width = w
        self.height = h
        self.ctx.viewport = (0, 0, w, h)

    def set_view(self, center: tuple = None, scale: float = None):
        """Set view transform (pan and zoom)."""
        if center is not None:
            self.view_center = center
        if scale is not None:
            self.view_scale = max(0.1, min(10.0, scale))
    
    def set_render_mode(self, mode: int):
        """Set render mode (0=standard, 1=wave, 2=energy, 3=minimal)."""
        self.render_mode = mode
    
    def set_glow(self, intensity: float):
        """Set glow intensity (0.0 to 2.0)."""
        self.glow_intensity = max(0.0, min(2.0, intensity))
    
    def set_particle_scale(self, scale: float):
        """Set global particle size multiplier."""
        self.particle_scale = max(0.1, min(5.0, scale))

    def _ensure_palette(self, num_types: int):
        """Lazily initialize or update color palette."""
        if self.palette is None or self.palette_size != num_types:
            from ..utils.colors import generate_hsv_palette
            pal_list = generate_hsv_palette(num_types)
            pal_int = np.array(pal_list, dtype=np.float32)
            self.palette = pal_int / 255.0
            self.palette_size = num_types

    def render(self, pos, vel, colors, angle, species_params, active_count, fps):
        """
        Render particles to the framebuffer.
        
        Args:
            pos: (N, 2) particle positions in world coords [-1, 1]
            vel: (N, 2) particle velocities (used for energy visualization)
            colors: (N,) particle type indices
            angle: (N,) particle rotation angles (radians)
            species_params: (T, 3) species parameters [radius, freq, amp]
            active_count: Number of active particles
            fps: Current FPS (for display/debug)
        """
        # Clear to ambient background
        self.ctx.clear(*self.ambient_color)
        
        if active_count <= 0:
            return
        
        # Ensure palette is up to date
        num_types = species_params.shape[0]
        self._ensure_palette(num_types)
        
        # Map particle types to colors (with bounds checking)
        color_indices = np.clip(colors[:active_count], 0, num_types - 1)
        p_colors = self.palette[color_indices]
        p_params = species_params[color_indices]
        
        # Calculate speed for energy visualization
        vel_slice = vel[:active_count]
        speeds = np.linalg.norm(vel_slice, axis=1).astype(np.float32)
        
        # Pack vertex data: pos(2) + color(3) + radius(1) + freq(1) + amp(1) + angle(1) + energy(1)
        data = np.empty((active_count, 10), dtype=np.float32)
        data[:, 0:2] = pos[:active_count]
        data[:, 2:5] = p_colors
        data[:, 5] = p_params[:, 0]  # radius
        data[:, 6] = p_params[:, 1]  # freq
        data[:, 7] = p_params[:, 2]  # amp
        data[:, 8] = angle[:active_count]
        data[:, 9] = speeds * 2.0  # Energy scale factor
        
        # Upload to GPU
        self.vbo.write(data.tobytes())
        
        # Calculate animation time
        current_time = time_module.time() - self.start_time
        
        # Set all uniforms
        self._set_uniform('window_size', (float(self.width), float(self.height)))
        self._set_uniform('view_center', self.view_center)
        self._set_uniform('view_scale', float(self.view_scale))
        self._set_uniform('time', float(current_time))
        self._set_uniform('particle_scale', float(self.particle_scale))
        self._set_uniform('render_mode', self.render_mode)
        self._set_uniform('glow_intensity', float(self.glow_intensity))
        self._set_uniform('edge_sharpness', float(self.edge_sharpness))
        self._set_uniform('ambient_color', self.ambient_color)
        self._set_uniform('wave_viz_mode', 0)
        
        # Render particles as points (geometry shader expands to quads)
        self.vao.render(moderngl.POINTS, vertices=active_count)

    def render_ui(self, surface: pygame.Surface):
        """
        Render pygame UI surface as overlay.
        
        Args:
            surface: Pygame surface with UI elements (RGBA)
        """
        # Get raw pixel data
        data = surface.get_view('1')
        
        # Create or update texture
        if self.ui_tex is None or self.ui_tex.size != surface.get_size():
            if self.ui_tex:
                self.ui_tex.release()
            self.ui_tex = self.ctx.texture(surface.get_size(), 4, data=data)
        else:
            self.ui_tex.write(data)
        
        # Render UI quad with texture
        self.ui_tex.use(0)
        self.ui_prog['ui_texture'].value = 0
        self.ctx.enable(moderngl.BLEND)
        self.ui_vao.render(moderngl.TRIANGLE_STRIP)
    
    def get_stats(self) -> dict:
        """Get renderer statistics."""
        return {
            'width': self.width,
            'height': self.height,
            'buffer_size': self.buffer_size,
            'palette_size': self.palette_size,
            'view_center': self.view_center,
            'view_scale': self.view_scale,
            'render_mode': self.render_mode,
            'glow_intensity': self.glow_intensity,
            'particle_scale': self.particle_scale,
        }