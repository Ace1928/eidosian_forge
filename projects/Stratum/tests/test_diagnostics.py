"""
Tests for the diagnostics module.
"""

import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.diagnostics import (
    Diagnostics,
    DiagnosticLevel,
    InvariantViolation,
    FieldStats,
    clamp_non_negative,
    clamp_to_finite,
    check_stability_bounds
)


class TestDiagnostics(unittest.TestCase):
    """Tests for the Diagnostics class."""

    def test_check_field_nan_detection(self):
        """Test NaN detection in fields."""
        diag = Diagnostics(level=DiagnosticLevel.BASIC, fail_fast=False)
        
        field = np.array([[1.0, 2.0], [np.nan, 4.0]])
        result = diag.check_field(field, "test", tick=1)
        
        self.assertFalse(result)
        self.assertEqual(len(diag.issues), 1)
        self.assertIn("NaN", diag.issues[0])

    def test_check_field_inf_detection(self):
        """Test Inf detection in fields."""
        diag = Diagnostics(level=DiagnosticLevel.BASIC, fail_fast=False)
        
        field = np.array([[1.0, np.inf], [3.0, 4.0]])
        result = diag.check_field(field, "test", tick=1)
        
        self.assertFalse(result)
        self.assertEqual(len(diag.issues), 1)
        self.assertIn("Inf", diag.issues[0])

    def test_warn_on_issue_emits_warning(self):
        """Test that warn_on_issue emits warnings for non-fatal checks."""
        diag = Diagnostics(level=DiagnosticLevel.BASIC, fail_fast=False, warn_on_issue=True)
        field = np.array([[1.0, np.inf], [3.0, 4.0]])
        with self.assertWarns(UserWarning):
            diag.check_field(field, "test", tick=1)

    def test_check_field_non_negative(self):
        """Test non-negative constraint checking."""
        diag = Diagnostics(level=DiagnosticLevel.STANDARD, fail_fast=False)
        
        field = np.array([[1.0, 2.0], [-0.5, 4.0]])
        result = diag.check_field(field, "test", tick=1, non_negative=True)
        
        self.assertFalse(result)
        self.assertIn("Negative", diag.issues[0])

    def test_check_field_clean_passes(self):
        """Test that clean fields pass all checks."""
        diag = Diagnostics(level=DiagnosticLevel.STRICT, fail_fast=True)
        
        field = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = diag.check_field(field, "test", tick=1, non_negative=True)
        
        self.assertTrue(result)

    def test_fail_fast_raises_exception(self):
        """Test that fail_fast mode raises exception."""
        diag = Diagnostics(level=DiagnosticLevel.BASIC, fail_fast=True)
        
        field = np.array([[1.0, np.nan], [3.0, 4.0]])
        
        with self.assertRaises(InvariantViolation) as ctx:
            diag.check_field(field, "test_field", tick=5)
        
        self.assertEqual(ctx.exception.field_name, "test_field")
        self.assertEqual(ctx.exception.tick, 5)

    def test_level_none_skips_checks(self):
        """Test that NONE level skips all checks."""
        diag = Diagnostics(level=DiagnosticLevel.NONE, fail_fast=True)
        
        # Even NaN should pass with NONE level
        field = np.array([[np.nan, np.inf], [-1.0, 4.0]])
        result = diag.check_field(field, "test", tick=1, non_negative=True)
        
        self.assertTrue(result)

    def test_compute_field_stats(self):
        """Test field statistics computation."""
        diag = Diagnostics()
        
        field = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats = diag.compute_field_stats(field, "test")
        
        self.assertEqual(stats.name, "test")
        self.assertEqual(stats.min_val, 1.0)
        self.assertEqual(stats.max_val, 4.0)
        self.assertAlmostEqual(stats.mean_val, 2.5)
        self.assertEqual(stats.nan_count, 0)
        self.assertEqual(stats.inf_count, 0)
        self.assertEqual(stats.negative_count, 0)

    def test_compute_field_stats_with_nan(self):
        """Test field statistics with NaN values."""
        diag = Diagnostics()
        
        field = np.array([[1.0, np.nan], [3.0, 4.0]])
        stats = diag.compute_field_stats(field, "test")
        
        self.assertEqual(stats.nan_count, 1)
        self.assertEqual(stats.min_val, 1.0)  # NaN excluded


class TestClampFunctions(unittest.TestCase):
    """Tests for clamping utility functions."""

    def test_clamp_non_negative(self):
        """Test clamping negative values."""
        field = np.array([[1.0, -2.0], [-3.0, 4.0]])
        loss = clamp_non_negative(field, record_loss=True)
        
        self.assertAlmostEqual(loss, 5.0)  # -2 + -3 = -5, abs = 5
        self.assertTrue(np.all(field >= 0))

    def test_clamp_non_negative_no_negatives(self):
        """Test clamping when there are no negatives."""
        field = np.array([[1.0, 2.0], [3.0, 4.0]])
        loss = clamp_non_negative(field)
        
        self.assertEqual(loss, 0.0)

    def test_clamp_to_finite(self):
        """Test replacing non-finite values."""
        field = np.array([[1.0, np.nan], [np.inf, 4.0]])
        count = clamp_to_finite(field, default=0.0)
        
        self.assertEqual(count, 2)
        self.assertTrue(np.all(np.isfinite(field)))

    def test_clamp_to_finite_all_finite(self):
        """Test that all-finite arrays are unchanged."""
        field = np.array([[1.0, 2.0], [3.0, 4.0]])
        original = field.copy()
        count = clamp_to_finite(field)
        
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(field, original)


class TestStabilityBounds(unittest.TestCase):
    """Tests for stability bound checking."""

    def test_stable_parameters(self):
        """Test stable parameter set."""
        is_stable, msg = check_stability_bounds(
            max_velocity=0.5,
            diffusivity=0.1,
            dt=1.0,
            dx=1.0
        )
        
        self.assertTrue(is_stable)
        self.assertIn("satisfied", msg)

    def test_advection_cfl_violation(self):
        """Test detection of CFL violation."""
        is_stable, msg = check_stability_bounds(
            max_velocity=2.0,  # v*dt/dx = 2 > 1
            diffusivity=0.1,
            dt=1.0,
            dx=1.0
        )
        
        self.assertFalse(is_stable)
        self.assertIn("Advection", msg)

    def test_diffusion_stability_violation(self):
        """Test detection of diffusion instability."""
        is_stable, msg = check_stability_bounds(
            max_velocity=0.1,
            diffusivity=1.0,  # D*dt/dx^2 = 1 > 0.5
            dt=1.0,
            dx=1.0
        )
        
        self.assertFalse(is_stable)
        self.assertIn("Diffusion", msg)


class TestInvariantViolation(unittest.TestCase):
    """Tests for the InvariantViolation exception."""

    def test_exception_message(self):
        """Test exception message formatting."""
        exc = InvariantViolation(
            message="Test violation",
            field_name="rho",
            tick=100,
            locations=[(0, 0), (1, 1)],
            values=[-0.5, -0.3]
        )
        
        msg = str(exc)
        self.assertIn("Test violation", msg)
        self.assertIn("rho", msg)
        self.assertIn("100", msg)

    def test_exception_with_context(self):
        """Test exception with additional context."""
        exc = InvariantViolation(
            message="Test",
            field_name="E_heat",
            tick=50,
            context={'source': 'advection'}
        )
        
        self.assertEqual(exc.context['source'], 'advection')


if __name__ == "__main__":
    unittest.main()
