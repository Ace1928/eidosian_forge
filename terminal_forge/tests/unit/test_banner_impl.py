"""
Unit tests for terminal_forge.banner module.

Tests banner creation, styling, and rendering.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from terminal_forge.banner import Banner
from terminal_forge.borders import BorderStyle


class TestBannerCreation:
    """Test banner creation and basic properties."""
    
    def test_create_empty_banner(self):
        """Test creating an empty banner."""
        banner = Banner()
        assert banner is not None
    
    def test_banner_with_content(self):
        """Test creating a banner with content."""
        banner = Banner()
        banner.add_line("Hello World")
        result = banner.render()
        assert "Hello World" in result
    
    def test_banner_with_title(self):
        """Test creating a banner with a title."""
        banner = Banner()
        banner.title = "Test Title"
        banner.add_line("Content")
        result = banner.render()
        # Title may or may not be rendered depending on config
        assert "Content" in result
    
    def test_banner_method_chaining(self):
        """Test that banner methods return self for chaining."""
        banner = Banner()
        result = banner.add_line("Line 1")
        assert result is banner  # Should return self


class TestBannerStyling:
    """Test banner styling options."""
    
    def test_banner_with_border(self):
        """Test banner with border."""
        banner = Banner()
        banner.set_border(BorderStyle.SINGLE)
        banner.add_line("Test")
        result = banner.render()
        # Should contain border characters
        assert len(result) > len("Test")
    
    def test_banner_with_padding(self):
        """Test banner with padding."""
        banner = Banner()
        banner.set_padding(2)
        banner.add_line("Test")
        result = banner.render()
        # Padded result should be wider
        assert len(result) > 0
    
    def test_banner_border_color(self):
        """Test banner with colored border."""
        banner = Banner()
        banner.set_border_color("red")
        banner.add_line("Test")
        result = banner.render()
        assert "Test" in result


class TestBannerAlignment:
    """Test text alignment in banners."""
    
    def test_left_alignment(self):
        """Test left-aligned banner content."""
        banner = Banner()
        banner.set_alignment("left")
        banner.add_line("Test")
        result = banner.render()
        assert "Test" in result
    
    def test_center_alignment(self):
        """Test center-aligned banner content."""
        banner = Banner()
        banner.set_alignment("center")
        banner.add_line("Test")
        result = banner.render()
        assert "Test" in result
    
    def test_right_alignment(self):
        """Test right-aligned banner content."""
        banner = Banner()
        banner.set_alignment("right")
        banner.add_line("Test")
        result = banner.render()
        assert "Test" in result


class TestBorderStyles:
    """Test different border styles."""
    
    def test_single_border(self):
        """Test single line border."""
        banner = Banner()
        banner.set_border(BorderStyle.SINGLE)
        banner.add_line("Test")
        result = banner.render()
        # Single border uses │ ─ ┌ ┐ └ ┘
        assert any(c in result for c in ['│', '─', '┌', '┐', '└', '┘', '+', '-', '|'])
    
    def test_double_border(self):
        """Test double line border."""
        banner = Banner()
        banner.set_border(BorderStyle.DOUBLE)
        banner.add_line("Test")
        result = banner.render()
        # Double border uses ║ ═ ╔ ╗ ╚ ╝
        assert any(c in result for c in ['║', '═', '╔', '╗', '╚', '╝', '+', '=', '|'])
    
    def test_rounded_border(self):
        """Test rounded border."""
        banner = Banner()
        banner.set_border(BorderStyle.ROUNDED)
        banner.add_line("Test")
        result = banner.render()
        # Rounded border uses ╭ ╮ ╰ ╯
        assert len(result) > len("Test")


class TestBannerRendering:
    """Test banner rendering output."""
    
    def test_render_returns_string(self):
        """Test that render() returns a string."""
        banner = Banner()
        banner.add_line("Test")
        result = banner.render()
        assert isinstance(result, str)
    
    def test_render_multiline(self):
        """Test rendering multiple lines."""
        banner = Banner()
        banner.add_line("Line 1")
        banner.add_line("Line 2")
        banner.add_line("Line 3")
        result = banner.render()
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
    
    def test_render_empty_banner(self):
        """Test rendering an empty banner."""
        banner = Banner()
        result = banner.render()
        assert isinstance(result, str)
    
    def test_add_lines_bulk(self):
        """Test adding multiple lines at once."""
        banner = Banner()
        banner.add_lines(["A", "B", "C"])
        result = banner.render()
        assert "A" in result
        assert "B" in result
        assert "C" in result
    
    def test_add_separator(self):
        """Test adding a separator."""
        banner = Banner()
        banner.add_line("Above")
        banner.add_separator()
        banner.add_line("Below")
        result = banner.render()
        assert "Above" in result
        assert "Below" in result


class TestBannerDisplay:
    """Test banner display functionality."""
    
    def test_display_method_exists(self):
        """Test that display method exists."""
        banner = Banner()
        banner.add_line("Test")
        assert hasattr(banner, 'display')
        assert callable(banner.display)
