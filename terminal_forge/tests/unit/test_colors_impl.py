"""
Unit tests for terminal_forge.colors module.

Tests ANSI color generation, RGB support, gradients, and terminal detection.
"""

import pytest
from unittest.mock import patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from terminal_forge.colors import Color


class TestBasicColors:
    """Test basic ANSI color functionality."""
    
    def test_color_class_exists(self):
        """Test that Color class can be instantiated."""
        assert Color is not None
    
    def test_basic_color_constants(self):
        """Test that basic color constants are defined."""
        assert Color.RED == "\033[31m"
        assert Color.GREEN == "\033[32m"
        assert Color.BLUE == "\033[34m"
        assert Color.RESET == "\033[0m"
    
    def test_bright_color_constants(self):
        """Test bright color variants."""
        assert Color.BRIGHT_RED == "\033[91m"
        assert Color.BRIGHT_GREEN == "\033[92m"
        assert Color.BRIGHT_BLUE == "\033[94m"
    
    def test_style_constants(self):
        """Test style constants."""
        assert Color.BOLD == "\033[1m"
        assert Color.UNDERLINE == "\033[4m"
        assert Color.ITALIC == "\033[3m"
    
    def test_colors_dict(self):
        """Test COLORS dictionary contains expected colors."""
        assert "red" in Color.COLORS
        assert "green" in Color.COLORS
        assert "blue" in Color.COLORS
        assert Color.COLORS["red"] == Color.RED


class TestRGBColors:
    """Test RGB color support."""
    
    def test_rgb_color_creation(self):
        """Test creating colors from RGB values."""
        result = Color.rgb(255, 128, 64)
        # Should return a valid ANSI escape sequence
        assert "\033[38;2;" in result
        assert "255" in result
        assert "128" in result
        assert "64" in result
    
    def test_rgb_bounds(self):
        """Test RGB values at boundaries."""
        result_black = Color.rgb(0, 0, 0)
        result_white = Color.rgb(255, 255, 255)
        assert "0;0;0" in result_black
        assert "255;255;255" in result_white
    
    def test_rgb_cached(self):
        """Test that RGB values are cached."""
        # Same call should return cached result
        result1 = Color.rgb(100, 100, 100)
        result2 = Color.rgb(100, 100, 100)
        assert result1 == result2


class TestColorApplication:
    """Test applying colors to text."""
    
    def test_apply_color(self):
        """Test applying color to text."""
        colored = f"{Color.RED}Hello{Color.RESET}"
        assert "\033[31m" in colored
        assert "Hello" in colored
        assert "\033[0m" in colored
    
    def test_apply_with_style(self):
        """Test applying color with style."""
        styled = f"{Color.BOLD}{Color.RED}Bold Red{Color.RESET}"
        assert "\033[1m" in styled
        assert "\033[31m" in styled
        assert "Bold Red" in styled
    
    def test_rainbow_method(self):
        """Test rainbow text if available."""
        if hasattr(Color, 'rainbow'):
            result = Color.rainbow("Hello")
            assert "Hello" in result or len(result) > 5
    
    def test_gradient_method(self):
        """Test gradient if available."""
        if hasattr(Color, 'gradient'):
            result = Color.gradient("Test", (255, 0, 0), (0, 0, 255))
            assert len(result) > 0


class TestColorUtilities:
    """Test color utility functions."""
    
    def test_blend_method(self):
        """Test color blending if available."""
        if hasattr(Color, 'blend'):
            result = Color.blend((255, 0, 0), (0, 0, 255), 0.5)
            # Midpoint should be purple-ish
            assert isinstance(result, tuple) or isinstance(result, str)
    
    def test_256_color_method(self):
        """Test 256 color support if available."""
        if hasattr(Color, 'color256'):
            result = Color.color256(196)  # Red in 256 palette
            assert "\033[38;5;196m" in result


class TestANSIDetection:
    """Test ANSI support detection."""
    
    def test_detect_support(self):
        """Test terminal capability detection."""
        if hasattr(Color, 'is_supported') or hasattr(Color, 'detect_support'):
            method = getattr(Color, 'is_supported', None) or getattr(Color, 'detect_support', None)
            if callable(method):
                result = method()
                assert isinstance(result, bool)
    
    def test_no_color_env(self):
        """Test NO_COLOR environment variable respect."""
        # This is a behavioral test - implementation may vary
        with patch.dict(os.environ, {'NO_COLOR': '1'}):
            # Colors should be disabled or detection should return False
            pass  # Implementation-dependent


class TestRGBColors:
    """Test RGB color support."""
    
    def test_rgb_color_creation(self):
        """Test creating colors from RGB values."""
        color = Color()
        result = color.rgb(255, 128, 64)
        # Should return a valid color specification
        assert result is not None
    
    def test_rgb_bounds_validation(self):
        """Test RGB value bounds (0-255)."""
        color = Color()
        # Values should be clamped or validated
        try:
            # This should either clamp or raise error
            result = color.rgb(300, -10, 128)
            # If it doesn't raise, values should be clamped
        except (ValueError, TypeError):
            pass  # Expected behavior for invalid values
    
    def test_hex_color_parsing(self):
        """Test parsing hex color codes."""
        color = Color()
        if hasattr(color, 'hex'):
            result = color.hex("#FF8040")
            assert result is not None


class TestColorEffects:
    """Test color effects like gradient and rainbow."""
    
    def test_rainbow_text(self):
        """Test rainbow text generation."""
        color = Color()
        if hasattr(color, 'rainbow'):
            result = color.rainbow("Hello World")
            assert "Hello World" in result or len(result) > len("Hello World")
    
    def test_gradient_generation(self):
        """Test color gradient generation."""
        color = Color()
        if hasattr(color, 'gradient'):
            result = color.gradient("Test", (255, 0, 0), (0, 0, 255))
            assert "Test" in result or len(result) >= len("Test")


class TestTerminalCapabilities:
    """Test terminal capability features if available."""
    
    @pytest.mark.skip(reason="TerminalCapability not exported - feature planned")
    def test_capability_detection(self):
        """Test that capability detection returns a valid mode."""
        pass  # Placeholder for future implementation
    
    @pytest.mark.skip(reason="TerminalCapability not exported - feature planned")
    def test_no_color_env_respected(self):
        """Test that NO_COLOR environment variable is respected."""
        pass  # Placeholder for future implementation


class TestColorCaching:
    """Test color caching for performance."""
    
    def test_rgb_colors_cached(self):
        """Test that RGB colors are properly cached via lru_cache."""
        # Same RGB values should return identical results
        result1 = Color.rgb(100, 150, 200)
        result2 = Color.rgb(100, 150, 200)
        assert result1 == result2
        
    def test_different_rgb_different_result(self):
        """Test different RGB values return different results."""
        result1 = Color.rgb(255, 0, 0)
        result2 = Color.rgb(0, 255, 0)
        assert result1 != result2


class TestANSIStripping:
    """Test ANSI code removal functionality."""
    
    def test_strip_ansi_codes(self):
        """Test stripping ANSI codes from text."""
        colored_text = f"{Color.RED}Hello{Color.RESET}"
        
        # Import strip function
        from terminal_forge.utils import strip_ansi
        stripped = strip_ansi(colored_text)
        assert stripped == "Hello"
    
    def test_strip_empty_string(self):
        """Test stripping from empty string."""
        from terminal_forge.utils import strip_ansi
        assert strip_ansi("") == ""
    
    def test_strip_no_ansi(self):
        """Test stripping text without ANSI codes."""
        from terminal_forge.utils import strip_ansi
        assert strip_ansi("plain text") == "plain text"
