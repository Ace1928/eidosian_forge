import unittest
import pygame.constants
def test_kscan__value_overlap(self):
    """Ensures no unexpected KSCAN constant values overlap."""
    EXPECTED_OVERLAPS = {frozenset(('KSCAN_' + n for n in item)) for item in K_AND_KSCAN_COMMON_OVERLAPS}
    overlaps = create_overlap_set(self.KSCAN_NAMES)
    self.assertSetEqual(overlaps, EXPECTED_OVERLAPS)