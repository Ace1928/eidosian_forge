"""
Optimized Rendering Layer.
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
        
        # Coordinate Mapping
        self.scale = self.width / 2.0
        self.offset_x = self.width / 2.0
        self.offset_y = self.height / 2.0
        
        # Resources
        self.sprites: List[pygame.Surface] = []
        self._init_resources()
        
    def _init_resources(self):
        """Generate sprites based on mode."""
        from ..utils.colors import generate_hsv_palette
        self.colors = generate_hsv_palette(self.cfg.num_types)
        
        # Calculate visual radius
        r_px = max(2, int(self.cfg.max_radius * 0.3 * self.scale))
        self.radius_px = r_px
        
        self.sprites = []
        for i in range(self.cfg.num_types):
            color = self.colors[i]
            
            if self.cfg.render_mode == RenderMode.GLOW:
                # Create glowing sprite
                s_size = r_px * 4
                surf = pygame.Surface((s_size, s_size), pygame.SRCALPHA)
                # Core
                pygame.draw.circle(surf, color, (s_size//2, s_size//2), r_px)
                # Halo (simulated by drawing larger, transparent circles)
                # Pygame blending is slow for this. Just use simple alpha.
                pygame.draw.circle(surf, (*color, 50), (s_size//2, s_size//2), r_px*2)
                self.sprites.append(surf)
                
            else:
                # Standard
                surf = pygame.Surface((r_px*2, r_px*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (r_px, r_px), r_px)
                self.sprites.append(surf)

    def resize(self, w, h):
        self.width = w
        self.height = h
        self.scale = w / 2.0
        self.offset_x = w / 2.0
        self.offset_y = h / 2.0
        # Re-init sprites if scale changed significantly?
        # For now, keep fixed size relative to world?
        # Actually, if window resizes, particles should scale or view should zoom?
        # Let's keep physics space fixed [-1, 1], so scale changes.
        self._init_resources()

    def render(self, pos, colors, active_count, fps):
        self.screen.fill((10, 10, 15))
        
        # Coordinate Transform
        # Use numpy for batch transform
        # We only process active particles
        
        p_active = pos[:active_count]
        c_active = colors[:active_count]
        
        # (x * scale) + offset
        sx = (p_active[:, 0] * self.scale) + self.offset_x
        # height - ((y * scale) + offset)
        sy = self.height - ((p_active[:, 1] * self.scale) + self.offset_y)
        
        # Center sprite
        if self.cfg.render_mode == RenderMode.GLOW:
            offset = self.radius_px * 2
        else:
            offset = self.radius_px
            
        sx -= offset
        sy -= offset
        
        # Batch Blit
        # Pygame blits sequence: ((surf, (x, y)), ...)
        # Construct list
        
        # Group by color for potential optimization (fewer surface switches?) 
        # Actually Pygame blits handles list.
        # Constructing the list is the Python bottleneck.
        
        if active_count > 10000:
            # Fallback to pixels for extreme counts
            # Use PixelArray?
            # Or just single pixel rects.
            self._render_fast_points(sx, sy, c_active)
        else:
            self._render_sprites(sx, sy, c_active)
            
        # HUD
        self._render_hud(fps, active_count)
        
        # Note: Flip handled by Caller or GUI Manager? 
        # Usually canvas just draws content.
        
    def _render_sprites(self, sx, sy, colors):
        blit_seq = []
        # Optimization: Use list comprehension with zip
        # Is iterating 5000 times in Python slow? Yes.
        # But constructing tuple list is necessary for blits.
        
        # Faster: Iterate by type
        for t in range(self.cfg.num_types):
            mask = (colors == t)
            if not np.any(mask): continue
            
            sprite = self.sprites[t]
            xs = sx[mask]
            ys = sy[mask]
            
            # Create (sprite, (x, y)) tuples
            # Zip is fast iterator
            blit_seq.extend((sprite, (x, y)) for x, y in zip(xs, ys))
            
        self.screen.blits(blit_seq)

    def _render_fast_points(self, sx, sy, colors):
        """Draw 2x2 rects for massive counts."""
        # This is slower in loop, but faster for GPU/Surface if we use draw.rect?
        # No, draw.rect is slow in loop.
        # PixelArray is fastest for single pixels.
        
        try:
            px_array = pygame.PixelArray(self.screen)
            # Map colors to integer
            # colors is index. self.colors is (R,G,B).
            # Convert to mapped integer color for surface
            mapped_colors = [self.screen.map_rgb(c) for c in self.colors]
            
            # Need integer coords
            ix = sx.astype(np.int32) + self.radius_px # re-center
            iy = sy.astype(np.int32) + self.radius_px
            
            # Bounds check for PixelArray
            # Numpy mask?
            valid_x = (ix >= 0) & (ix < self.width)
            valid_y = (iy >= 0) & (iy < self.height)
            mask = valid_x & valid_y
            
            ix = ix[mask]
            iy = iy[mask]
            c_vals = colors[mask]
            
            # This loop is still Python.
            # Cython or Numba could write to a buffer?
            # For now, just a loop.
            for x, y, c in zip(ix, iy, c_vals):
                px_array[x, y] = mapped_colors[c]
                
            px_array.close()
        except Exception:
            pass

    def _render_hud(self, fps, count):
        txt = f"FPS: {fps:.1f} | Particles: {count} | Mode: {self.cfg.render_mode.value}"
        surf = self.font.render(txt, True, (200, 200, 200))
        self.screen.blit(surf, (10, self.height - 30))
