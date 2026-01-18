"""
Tests for the core.types module.

This module tests the fundamental type definitions including Vec2, Cell,
and utility functions like dot() and clamp().
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import Vec2, Cell, dot, clamp


class TestVec2(unittest.TestCase):
    """Tests for the Vec2 class."""

    def test_creation(self):
        """Test Vec2 creation with x and y components."""
        v = Vec2(3.0, 4.0)
        self.assertEqual(v.x, 3.0)
        self.assertEqual(v.y, 4.0)

    def test_addition(self):
        """Test Vec2 addition operator."""
        v1 = Vec2(1.0, 2.0)
        v2 = Vec2(3.0, 4.0)
        result = v1 + v2
        self.assertEqual(result.x, 4.0)
        self.assertEqual(result.y, 6.0)

    def test_subtraction(self):
        """Test Vec2 subtraction operator."""
        v1 = Vec2(5.0, 6.0)
        v2 = Vec2(2.0, 3.0)
        result = v1 - v2
        self.assertEqual(result.x, 3.0)
        self.assertEqual(result.y, 3.0)

    def test_multiplication(self):
        """Test Vec2 scalar multiplication."""
        v = Vec2(2.0, 3.0)
        result = v * 2.5
        self.assertEqual(result.x, 5.0)
        self.assertEqual(result.y, 7.5)

    def test_rmul(self):
        """Test Vec2 reverse scalar multiplication."""
        v = Vec2(2.0, 3.0)
        result = 2.5 * v
        self.assertEqual(result.x, 5.0)
        self.assertEqual(result.y, 7.5)

    def test_division(self):
        """Test Vec2 scalar division."""
        v = Vec2(6.0, 9.0)
        result = v / 3.0
        self.assertEqual(result.x, 2.0)
        self.assertEqual(result.y, 3.0)

    def test_division_by_zero(self):
        """Test Vec2 division by zero returns zero vector."""
        v = Vec2(6.0, 9.0)
        result = v / 0
        self.assertEqual(result.x, 0.0)
        self.assertEqual(result.y, 0.0)

    def test_length(self):
        """Test Vec2 length calculation."""
        v = Vec2(3.0, 4.0)
        self.assertAlmostEqual(v.length(), 5.0, places=10)

    def test_length_zero_vector(self):
        """Test Vec2 length of zero vector."""
        v = Vec2(0.0, 0.0)
        self.assertEqual(v.length(), 0.0)

    def test_normalized(self):
        """Test Vec2 normalization."""
        v = Vec2(3.0, 4.0)
        n = v.normalized()
        self.assertAlmostEqual(n.x, 0.6, places=10)
        self.assertAlmostEqual(n.y, 0.8, places=10)
        self.assertAlmostEqual(n.length(), 1.0, places=10)

    def test_normalized_zero_vector(self):
        """Test Vec2 normalization of zero vector returns zero vector."""
        v = Vec2(0.0, 0.0)
        n = v.normalized()
        self.assertEqual(n.x, 0.0)
        self.assertEqual(n.y, 0.0)


class TestCell(unittest.TestCase):
    """Tests for the Cell class."""

    def test_creation(self):
        """Test Cell creation with i and j indices."""
        c = Cell(5, 10)
        self.assertEqual(c.i, 5)
        self.assertEqual(c.j, 10)

    def test_cell_is_frozen(self):
        """Test that Cell is immutable (frozen dataclass)."""
        c = Cell(1, 2)
        with self.assertRaises(AttributeError):
            c.i = 5

    def test_cell_hashable(self):
        """Test that Cell can be used as dictionary key."""
        c1 = Cell(1, 2)
        c2 = Cell(1, 2)
        c3 = Cell(3, 4)
        d = {c1: "a", c3: "b"}
        self.assertEqual(d[c2], "a")
        self.assertEqual(d[c3], "b")

    def test_neighbors4_interior(self):
        """Test Cell.neighbors4 for an interior cell."""
        c = Cell(5, 5)
        neighbors = c.neighbors4(10, 10)
        self.assertEqual(len(neighbors), 4)
        expected = {Cell(4, 5), Cell(6, 5), Cell(5, 4), Cell(5, 6)}
        self.assertEqual(set(neighbors), expected)

    def test_neighbors4_corner(self):
        """Test Cell.neighbors4 for a corner cell."""
        c = Cell(0, 0)
        neighbors = c.neighbors4(10, 10)
        self.assertEqual(len(neighbors), 2)
        expected = {Cell(1, 0), Cell(0, 1)}
        self.assertEqual(set(neighbors), expected)

    def test_neighbors4_edge(self):
        """Test Cell.neighbors4 for an edge cell."""
        c = Cell(0, 5)
        neighbors = c.neighbors4(10, 10)
        self.assertEqual(len(neighbors), 3)
        expected = {Cell(1, 5), Cell(0, 4), Cell(0, 6)}
        self.assertEqual(set(neighbors), expected)


class TestDotFunction(unittest.TestCase):
    """Tests for the dot product function."""

    def test_dot_product(self):
        """Test dot product of two vectors."""
        v1 = Vec2(1.0, 2.0)
        v2 = Vec2(3.0, 4.0)
        result = dot(v1, v2)
        self.assertEqual(result, 11.0)  # 1*3 + 2*4

    def test_dot_product_perpendicular(self):
        """Test dot product of perpendicular vectors is zero."""
        v1 = Vec2(1.0, 0.0)
        v2 = Vec2(0.0, 1.0)
        result = dot(v1, v2)
        self.assertEqual(result, 0.0)

    def test_dot_product_parallel(self):
        """Test dot product of parallel vectors."""
        v1 = Vec2(2.0, 0.0)
        v2 = Vec2(3.0, 0.0)
        result = dot(v1, v2)
        self.assertEqual(result, 6.0)

    def test_dot_product_negative(self):
        """Test dot product with negative components."""
        v1 = Vec2(-1.0, 2.0)
        v2 = Vec2(3.0, -4.0)
        result = dot(v1, v2)
        self.assertEqual(result, -11.0)  # -1*3 + 2*(-4)


class TestClampFunction(unittest.TestCase):
    """Tests for the clamp utility function."""

    def test_clamp_within_range(self):
        """Test clamp with value within range."""
        result = clamp(5.0, 0.0, 10.0)
        self.assertEqual(result, 5.0)

    def test_clamp_below_min(self):
        """Test clamp with value below minimum."""
        result = clamp(-5.0, 0.0, 10.0)
        self.assertEqual(result, 0.0)

    def test_clamp_above_max(self):
        """Test clamp with value above maximum."""
        result = clamp(15.0, 0.0, 10.0)
        self.assertEqual(result, 10.0)

    def test_clamp_at_min(self):
        """Test clamp with value at minimum."""
        result = clamp(0.0, 0.0, 10.0)
        self.assertEqual(result, 0.0)

    def test_clamp_at_max(self):
        """Test clamp with value at maximum."""
        result = clamp(10.0, 0.0, 10.0)
        self.assertEqual(result, 10.0)

    def test_clamp_negative_range(self):
        """Test clamp with negative range values."""
        result = clamp(-5.0, -10.0, -1.0)
        self.assertEqual(result, -5.0)
        result = clamp(-15.0, -10.0, -1.0)
        self.assertEqual(result, -10.0)


if __name__ == "__main__":
    unittest.main()
