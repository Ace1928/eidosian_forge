import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def test_surface_to_array_2d(self):
    try:
        from numpy import empty, dtype
    except ImportError:
        return
    palette = self.test_palette
    alpha_color = (0, 0, 0, 128)
    dst_dims = self.surf_size
    destinations = [empty(dst_dims, t) for t in self.dst_types]
    if pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN:
        swapped_dst = empty(dst_dims, dtype('>u4'))
    else:
        swapped_dst = empty(dst_dims, dtype('<u4'))
    for surf in self.sources:
        src_bytesize = surf.get_bytesize()
        for dst in destinations:
            if dst.itemsize < src_bytesize:
                self.assertRaises(ValueError, surface_to_array, dst, surf)
                continue
            dst[...] = 0
            self.assertFalse(surf.get_locked())
            surface_to_array(dst, surf)
            self.assertFalse(surf.get_locked())
            for posn, i in self.test_points:
                sp = unsigned32(surf.get_at_mapped(posn))
                dp = dst[posn]
                self.assertEqual(dp, sp, '%s != %s: flags: %i, bpp: %i, dtype: %s,  posn: %s' % (dp, sp, surf.get_flags(), surf.get_bitsize(), dst.dtype, posn))
            if surf.get_masks()[3]:
                posn = (2, 1)
                surf.set_at(posn, alpha_color)
                surface_to_array(dst, surf)
                sp = unsigned32(surf.get_at_mapped(posn))
                dp = dst[posn]
                self.assertEqual(dp, sp, '%s != %s: bpp: %i' % (dp, sp, surf.get_bitsize()))
        swapped_dst[...] = 0
        self.assertFalse(surf.get_locked())
        surface_to_array(swapped_dst, surf)
        self.assertFalse(surf.get_locked())
        for posn, i in self.test_points:
            sp = unsigned32(surf.get_at_mapped(posn))
            dp = swapped_dst[posn]
            self.assertEqual(dp, sp, '%s != %s: flags: %i, bpp: %i, dtype: %s,  posn: %s' % (dp, sp, surf.get_flags(), surf.get_bitsize(), dst.dtype, posn))
        if surf.get_masks()[3]:
            posn = (2, 1)
            surf.set_at(posn, alpha_color)
            self.assertFalse(surf.get_locked())
            surface_to_array(swapped_dst, surf)
            self.assertFalse(surf.get_locked())
            sp = unsigned32(surf.get_at_mapped(posn))
            dp = swapped_dst[posn]
            self.assertEqual(dp, sp, '%s != %s: bpp: %i' % (dp, sp, surf.get_bitsize()))