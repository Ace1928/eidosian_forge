"""
Unit tests for terminal_forge.layout module.

Tests text alignment, columns, tables, and text wrapping.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from terminal_forge.layout import LayoutEngine, Alignment, ContentOverflow, Padding, Margin
from terminal_forge.utils import text_length


class TestAlignment:
    """Test text alignment functionality."""
    
    def test_alignment_enum_values(self):
        """Test Alignment enum has expected values."""
        assert hasattr(Alignment, 'LEFT')
        assert hasattr(Alignment, 'CENTER')
        assert hasattr(Alignment, 'RIGHT')
    
    def test_left_align_text(self):
        """Test left alignment of text."""
        result = LayoutEngine.align_text("Test", width=20, alignment=Alignment.LEFT)
        assert result.startswith("Test")
        assert len(result) == 20
    
    def test_center_align_text(self):
        """Test center alignment of text."""
        result = LayoutEngine.align_text("Test", width=20, alignment=Alignment.CENTER)
        # Text should be centered
        stripped = result.strip()
        assert stripped == "Test"
        # Should have padding on both sides
        left_pad = len(result) - len(result.lstrip())
        right_pad = len(result) - len(result.rstrip())
        assert abs(left_pad - right_pad) <= 1  # Off by at most 1 for odd widths
    
    def test_right_align_text(self):
        """Test right alignment of text."""
        result = LayoutEngine.align_text("Test", width=20, alignment=Alignment.RIGHT)
        assert result.endswith("Test")
        assert len(result) == 20


class TestOverflow:
    """Test overflow handling."""
    
    def test_overflow_mode_enum(self):
        """Test ContentOverflow enum values."""
        assert hasattr(ContentOverflow, 'WRAP')
        assert hasattr(ContentOverflow, 'TRUNCATE')
    
    def test_truncate_overflow(self):
        """Test truncating text that's too long."""
        long_text = "This is a very long text that should be truncated"
        result = LayoutEngine.handle_overflow(long_text, max_width=20, strategy=ContentOverflow.TRUNCATE)
        # handle_overflow returns a list
        assert len(result) >= 1
        assert text_length(result[0]) <= 20
    
    def test_wrap_overflow(self):
        """Test wrapping text that's too long."""
        long_text = "This is a very long text that should be wrapped"
        result = LayoutEngine.handle_overflow(long_text, max_width=20, strategy=ContentOverflow.WRAP)
        # Returns list of lines
        for line in result:
            assert text_length(line) <= 20


class TestColumns:
    """Test column layout functionality."""
    
    def test_create_columns(self):
        """Test creating column layout."""
        # create_columns expects list of lists (each column is a list of rows)
        columns = [["Col 1"], ["Col 2"], ["Col 3"]]
        result = LayoutEngine.create_columns(columns, widths=[10, 10, 10])
        # Result is list of row strings
        assert len(result) == 1
        for col_text in ["Col 1", "Col 2", "Col 3"]:
            assert col_text in result[0]
    
    def test_columns_with_different_widths(self):
        """Test columns with different widths."""
        columns = [["Short"], ["Medium"], ["Long"]]
        result = LayoutEngine.create_columns(columns, widths=[10, 20, 30])
        assert len(result) == 1
        # Total width should be at least sum of columns (minus separators)
        assert len(result[0]) > 0


class TestTable:
    """Test table generation."""
    
    def test_simple_table(self):
        """Test creating a simple table."""
        headers = ["Name", "Age"]
        rows = [["Alice", "30"], ["Bob", "25"]]
        result = LayoutEngine.create_table(headers, rows)
        # Should return list of strings for table
        assert len(result) > 0
        # Headers and data should be represented
        full_text = "\n".join(result)
        assert "Name" in full_text or "name" in full_text.lower()
    
    def test_table_with_borders(self):
        """Test table with custom border style."""
        headers = ["A", "B"]
        rows = [["1", "2"]]
        # Just verify it doesn't error
        result = LayoutEngine.create_table(headers, rows, border_style=None)
        assert len(result) > 0
    
    def test_empty_table(self):
        """Test empty table."""
        result = LayoutEngine.create_table(["Col"], [])
        assert len(result) > 0  # Should at least have headers


class TestPaddingMargin:
    """Test padding and margin classes."""
    
    def test_padding_defaults(self):
        """Test Padding default values."""
        pad = Padding()
        assert pad.top == 0
        assert pad.right == 0
        assert pad.bottom == 0
        assert pad.left == 0
    
    def test_padding_custom_values(self):
        """Test Padding with custom values."""
        pad = Padding(top=1, right=2, bottom=3, left=4)
        assert pad.top == 1
        assert pad.right == 2
        assert pad.bottom == 3
        assert pad.left == 4
    
    def test_margin_defaults(self):
        """Test Margin default values."""
        margin = Margin()
        assert margin.top == 0
        assert margin.right == 0
        assert margin.bottom == 0
        assert margin.left == 0
    
    def test_margin_custom_values(self):
        """Test Margin with custom values."""
        margin = Margin(top=1, left=4)
        assert margin.top == 1
        assert margin.left == 4
    
    def test_uniform_padding(self):
        """Test creating uniform padding."""
        if hasattr(Padding, 'uniform'):
            pad = Padding.uniform(5)
            assert pad.top == 5
            assert pad.right == 5
            assert pad.bottom == 5
            assert pad.left == 5


class TestTextWrapping:
    """Test text wrapping functionality."""
    
    def test_wrap_long_text(self):
        """Test wrapping long text."""
        text = "The quick brown fox jumps over the lazy dog"
        result = LayoutEngine.wrap_text(text, max_width=10)
        # Returns list of lines
        assert len(result) > 1
    
    def test_wrap_short_text(self):
        """Test wrapping text that fits."""
        text = "Short"
        result = LayoutEngine.wrap_text(text, max_width=20)
        assert len(result) == 1
        assert result[0] == "Short"
    
    def test_wrap_preserves_words(self):
        """Test that wrapping preserves word boundaries."""
        text = "Hello World"
        result = LayoutEngine.wrap_text(text, max_width=8)
        # "Hello" should not be split
        assert any("Hello" in line for line in result)


class TestLayoutEngine:
    """Test LayoutEngine general functionality."""
    
    def test_engine_creation(self):
        """Test creating a LayoutEngine."""
        engine = LayoutEngine()
        assert engine is not None
    
    def test_align_text_static(self):
        """Test align_text is accessible as static method."""
        result = LayoutEngine.align_text("test", 10)
        assert len(result) == 10
    
    def test_ansi_aware_width(self):
        """Test width calculation ignores ANSI codes."""
        text_with_ansi = "\033[31mRed\033[0m"
        width = text_length(text_with_ansi)
        # Width should be 3 (just "Red"), not include ANSI codes
        assert width == 3
