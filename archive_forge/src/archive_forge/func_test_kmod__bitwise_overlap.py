import unittest
import pygame.constants
def test_kmod__bitwise_overlap(self):
    """Ensures certain KMOD constants have overlapping bits."""
    KMOD_COMPRISED_DICT = {'KMOD_SHIFT': ('KMOD_LSHIFT', 'KMOD_RSHIFT'), 'KMOD_CTRL': ('KMOD_LCTRL', 'KMOD_RCTRL'), 'KMOD_ALT': ('KMOD_LALT', 'KMOD_RALT'), 'KMOD_META': ('KMOD_LMETA', 'KMOD_RMETA'), 'KMOD_GUI': ('KMOD_LGUI', 'KMOD_RGUI')}
    for base_name, seq_names in KMOD_COMPRISED_DICT.items():
        expected_value = 0
        for name in seq_names:
            expected_value |= getattr(pygame.constants, name)
        value = getattr(pygame.constants, base_name)
        self.assertEqual(value, expected_value)