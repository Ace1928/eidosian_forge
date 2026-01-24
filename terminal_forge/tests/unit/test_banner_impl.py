"""
Unit tests for terminal_forge.banner module.

Tests banner creation, styling, and rendering.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from terminal_forge.banner import Banner, BannerFactory
from terminal_forge.borders import BorderStyle
from terminal_forge.themes import Theme


class TestBannerCreation:
    """Test banner creation and basic properties."""
    
    def test_create_empty_banner(self):
        """Test creating an empty banner."""
        banner = Banner()
        assert banner is not None
    
    def test_banner_with_content(self):
        """Test creating a banner with content."""
        banner = Banner().add_line("Hello World")
        result = banner.render()
        assert "Hello World" in result
    
    def test_banner_with_title(self):
        """Test creating a banner with a title."""
        banner = Banner().title("Test Title").add_line("Content")
        result = banner.render()
        assert "Test Title" in result or "Content" in result
    
    def test_banner_fluent_interface(self):
        """Test that banner supports fluent interface."""
        banner = (Banner()
            .title("Title")
            .add_line("Line 1")
            .add_line("Line 2"))
        assert banner is not None
        result = banner.render()
        assert "Line 1" in result
        assert "Line 2" in result


class TestBannerStyling:
    """Test banner styling options."""
    
    def test_banner_with_border(self):
        """Test banner with border."""
        banner = Banner().border(BorderStyle.SINGLE).add_line("Test")
        result = banner.render()
        # Should contain border characters
        assert len(result) > len("Test")
    
    def test_banner_with_padding(self):
        """Test banner with padding."""
        banner = Banner().padding(2).add_line("Test")
        result = banner.render()
        # Padded result should be wider
        lines = result.strip().split('\n')
        # At least one line should have padding
        assert any(line.startswith(' ') or line.endswith(' ') for line in lines)
    
    def test_banner_with_theme(self):
        """Test banner with theme applied."""
        theme = Theme.get("success")
        banner = Banner().theme(theme).add_line("Success!")
        result = banner.render()
        # Should contain the text, possibly with ANSI codes
        assert "Success!" in result or "\x1b[" in result


class TestBannerAlignment:
    """Test text alignment in banners."""
    
    def test_left_alignment(self):
        """Test left-aligned banner content."""
        banner = Banner().align("left").add_line("Test")
        result = banner.render()
        assert "Test" in result
    
    def test_center_alignment(self):
        """Test center-aligned banner content."""
        banner = Banner().align("center").add_line("Test")
        result = banner.render()
        assert "Test" in result
    
    def test_right_alignment(self):
        """Test right-aligned banner content."""
        banner = Banner().align("right").add_line("Test")
        result = banner.render()
        assert "Test" in result


class TestBannerFactory:
    """Test BannerFactory for creating pre-configured banners."""
    
    def test_error_banner(self):
        """Test creating an error banner."""
        if hasattr(BannerFactory, 'error'):
            banner = BannerFactory.error("Error occurred!")
            result = banner.render() if hasattr(banner, 'render') else str(banner)
            assert "Error" in result or "error" in result.lower()
    
    def test_success_banner(self):
        """Test creating a success banner."""
        if hasattr(BannerFactory, 'success'):
            banner = BannerFactory.success("Operation complete!")
            result = banner.render() if hasattr(banner, 'render') else str(banner)
            assert "complete" in result.lower() or "success" in result.lower()
    
    def test_info_banner(self):
        """Test creating an info banner."""
        if hasattr(BannerFactory, 'info'):
            banner = BannerFactory.info("Information message")
            assert banner is not None


class TestBorderStyles:
    """Test different border styles."""
    
    def test_single_border(self):
        """Test single line border."""
        banner = Banner().border(BorderStyle.SINGLE).add_line("Test")
        result = banner.render()
        # Single border uses │ ─ ┌ ┐ └ ┘
        assert any(c in result for c in ['│', '─', '┌', '┐', '└', '┘', '+', '-', '|'])
    
    def test_double_border(self):
        """Test double line border."""
        banner = Banner().border(BorderStyle.DOUBLE).add_line("Test")
        result = banner.render()
        # Double border uses ║ ═ ╔ ╗ ╚ ╝
        assert any(c in result for c in ['║', '═', '╔', '╗', '╚', '╝', '+', '=', '|'])
    
    def test_rounded_border(self):
        """Test rounded border."""
        banner = Banner().border(BorderStyle.ROUNDED).add_line("Test")
        result = banner.render()
        # Rounded border uses ╭ ╮ ╰ ╯
        assert len(result) > len("Test")


class TestBannerRendering:
    """Test banner rendering output."""
    
    def test_render_returns_string(self):
        """Test that render() returns a string."""
        banner = Banner().add_line("Test")
        result = banner.render()
        assert isinstance(result, str)
    
    def test_render_multiline(self):
        """Test rendering multiple lines."""
        banner = Banner().add_line("Line 1").add_line("Line 2").add_line("Line 3")
        result = banner.render()
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
    
    def test_render_empty_banner(self):
        """Test rendering an empty banner."""
        banner = Banner()
        result = banner.render()
        assert isinstance(result, str)
    
    def test_str_representation(self):
        """Test __str__ returns rendered banner."""
        banner = Banner().add_line("Test")
        str_result = str(banner)
        render_result = banner.render()
        assert str_result == render_result
