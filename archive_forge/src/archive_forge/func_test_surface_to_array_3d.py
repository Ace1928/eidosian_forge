import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def test_surface_to_array_3d(self):
    try:
        from numpy import empty, dtype
    except ImportError:
        return
    palette = self.test_palette
    dst_dims = self.surf_size + (3,)
    destinations = [empty(dst_dims, t) for t in self.dst_types]
    if pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN:
        swapped_dst = empty(dst_dims, dtype('>u4'))
    else:
        swapped_dst = empty(dst_dims, dtype('<u4'))
    for surf in self.sources:
        src_bitsize = surf.get_bitsize()
        for dst in destinations:
            dst[...] = 0
            self.assertFalse(surf.get_locked())
            surface_to_array(dst, surf)
            self.assertFalse(surf.get_locked())
            for posn, i in self.test_points:
                r_surf, g_surf, b_surf, a_surf = surf.get_at(posn)
                r_arr, g_arr, b_arr = dst[posn]
                self.assertEqual(r_arr, r_surf, '%i != %i, color: red, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))
                self.assertEqual(g_arr, g_surf, '%i != %i, color: green, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))
                self.assertEqual(b_arr, b_surf, '%i != %i, color: blue, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))
        swapped_dst[...] = 0
        self.assertFalse(surf.get_locked())
        surface_to_array(swapped_dst, surf)
        self.assertFalse(surf.get_locked())
        for posn, i in self.test_points:
            r_surf, g_surf, b_surf, a_surf = surf.get_at(posn)
            r_arr, g_arr, b_arr = swapped_dst[posn]
            self.assertEqual(r_arr, r_surf, '%i != %i, color: red, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))
            self.assertEqual(g_arr, g_surf, '%i != %i, color: green, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))
            self.assertEqual(b_arr, b_surf, '%i != %i, color: blue, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))