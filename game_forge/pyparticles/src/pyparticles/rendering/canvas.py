"""
Optimized Rendering Layer.
Supports Waves.
"""
import pygame
import numpy as np
from typing import List, Tuple
from ..core.types import SimulationConfig, RenderMode

class Canvas:
    def __init__(self, config: SimulationConfig, caption="Eidosian PyParticles"):
        pygame.init()
        self.cfg = config
        self.width = config.width
        self.height = config.height
        
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption(caption)
        
        self.font = pygame.font.SysFont("Consolas", 14)
        
        self.scale = self.width / 2.0
        self.offset_x = self.width / 2.0
        self.offset_y = self.height / 2.0
        
        self.sprites: List[pygame.Surface] = []
        self._init_resources()
        
    def _init_resources(self):
        """Generate sprites based on mode."""
        from ..utils.colors import generate_hsv_palette
        self.colors = generate_hsv_palette(self.cfg.num_types)
        
        # Calculate visual radius
        r_px = max(2, int(self.cfg.default_max_radius * 0.3 * self.scale))
        self.radius_px = r_px
        
        self.sprites = []
        for i in range(self.cfg.num_types):
            color = self.colors[i]
            
            if self.cfg.render_mode == RenderMode.GLOW:
                s_size = r_px * 4
                surf = pygame.Surface((s_size, s_size), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (s_size//2, s_size//2), r_px)
                pygame.draw.circle(surf, (*color, 50), (s_size//2, s_size//2), r_px*2)
                self.sprites.append(surf)
                
            else:
                # Standard or Wave (Wave uses dynamic drawing usually, but sprite fallback)
                surf = pygame.Surface((r_px*2, r_px*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (r_px, r_px), r_px)
                self.sprites.append(surf)

    def resize(self, w, h):
        self.width = w
        self.height = h
        self.scale = w / 2.0
        self.offset_x = w / 2.0
        self.offset_y = h / 2.0
        self._init_resources()

    def render(self, pos, colors, angle, species_params, active_count, fps):
        self.screen.fill((10, 10, 15))
        
        p_active = pos[:active_count]
        c_active = colors[:active_count]
        
        sx = (p_active[:, 0] * self.scale) + self.offset_x
        sy = self.height - ((p_active[:, 1] * self.scale) + self.offset_y)
        
        # Wave Rendering?
        if self.cfg.render_mode == RenderMode.WAVE and active_count < 2000:
            self._render_waves(sx, sy, c_active, angle[:active_count], species_params)
        elif active_count > 10000:
            self._render_fast_points(sx, sy, c_active)
        else:
            # Re-center for sprites
            if self.cfg.render_mode == RenderMode.GLOW:
                offset = self.radius_px * 2
            else:
                offset = self.radius_px
            self._render_sprites(sx - offset, sy - offset, c_active)
            
        self._render_hud(fps, active_count)
        
    def _render_sprites(self, sx, sy, colors):
        blit_seq = []
        for t in range(self.cfg.num_types):
            mask = (colors == t)
            if not np.any(mask): continue
            
            sprite = self.sprites[t]
            xs = sx[mask].astype(int)
            ys = sy[mask].astype(int)
            
            blit_seq.extend((sprite, (x, y)) for x, y in zip(xs, ys))
            
        self.screen.blits(blit_seq)

    def _render_fast_points(self, sx, sy, colors):
        try:
            px_array = pygame.PixelArray(self.screen)
            mapped_colors = [self.screen.map_rgb(c) for c in self.colors]
            
            ix = sx.astype(np.int32)
            iy = sy.astype(np.int32)
            
            valid_x = (ix >= 0) & (ix < self.width)
            valid_y = (iy >= 0) & (iy < self.height)
            mask = valid_x & valid_y
            
            ix = ix[mask]
            iy = iy[mask]
            c_vals = colors[mask]
            
            for x, y, c in zip(ix, iy, c_vals):
                px_array[x, y] = mapped_colors[c]
                
            px_array.close()
        except Exception:
            pass

    def _render_waves(self, sx, sy, colors, angles, species_params):
        """Draw approximated shapes."""
        # This is slow in Python loop. Use for low N or debugging.
        # Draw circle + line for orientation
        for i in range(len(colors)):
            t = colors[i]
            x, y = int(sx[i]), int(sy[i])
            angle = angles[i]
            
            rad = species_params[t, 0] * self.scale
            amp = species_params[t, 2] * self.scale
            freq = species_params[t, 1]
            
            color = self.colors[t]
            
            # Draw base
            pygame.draw.circle(self.screen, color, (x, y), max(1, int(rad)), 1)
            
            # Draw Orientation
            end_x = x + int(rad * np.cos(angle))
            end_y = y - int(rad * np.sin(angle)) # Y flipped
            pygame.draw.line(self.screen, (255,255,255), (x, y), (end_x, end_y))
            
            # Visualizing wave perimeter is computationally expensive here (polygon of 20+ points)
            # Just visualize phase peak?
            # Peak is at angle + 0?
            # cos(freq * theta). Peak at theta = 0.
            # Local theta = 0 means world angle = p_angle.
            # So the line indicates a Peak.

    def _render_hud(self, fps, count):
        txt = f"FPS: {fps:.1f} | Particles: {count} | Mode: {self.cfg.render_mode.value}"
        surf = self.font.render(txt, True, (200, 200, 200))
        self.screen.blit(surf, (10, self.height - 30))