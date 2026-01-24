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


class TestAlignment:
    """Test text alignment functionality."""
    
    def test_alignment_enum_values(self):
        """Test Alignment enum has expected values."""
        assert hasattr(Alignment, 'LEFT')
        assert hasattr(Alignment, 'CENTER')
        assert hasattr(Alignment, 'RIGHT')
    
    def test_left_align_text(self):
        """Test left alignment of text."""
        engine = LayoutEngine()
        result = engine.align("Test", width=20, alignment=Alignment.LEFT)
        assert result.startswith("Test")
        assert len(result) == 20
    
    def test_center_align_text(self):
        """Test center alignment of text."""
        engine = LayoutEngine()
        result = engine.align("Test", width=20, alignment=Alignment.CENTER)
        # Text should be centered (8 spaces on each side for "Test")
        stripped = result.strip()
        assert stripped == "Test"
        # Should have padding on both sides
        left_pad = len(result) - len(result.lstrip())
        right_pad = len(result) - len(result.rstrip())
        assert abs(left_pad - right_pad) <= 1  # Off by at most 1 for odd widths
    
    def test_right_align_text(self):
        """Test right alignment of text."""
        engine = LayoutEngine()
        result = engine.align("Test", width=20, alignment=Alignment.RIGHT)
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
        engine = LayoutEngine()
        long_text = "This is a very long text that should be truncated"
        result = engine.truncate(long_text, width=20)
        assert len(result) <= 20
    
    def test_wrap_overflow(self):
        """Test wrapping text that's too long."""
        engine = LayoutEngine()
        long_text = "This is a very long text that should be wrapped"
        result = engine.wrap(long_text, width=20)
        lines = result if isinstance(result, list) else result.split('\n')
        for line in lines:
            assert len(line.rstrip()) <= 20


class TestColumns:
    """Test column layout functionality."""
    
    def test_create_columns(self):
        """Test creating column layout."""
        engine = LayoutEngine()
        columns = ["Col 1", "Col 2", "Col 3"]
        result = engine.columns(columns, widths=[10, 10, 10])
        # All columns should be in result
        for col in columns:
            assert col in result or col.strip() in result
    
    def test_columns_with_different_widths(self):
        """Test columns with different widths."""
        engine = LayoutEngine()
        columns = ["Short", "Medium Length", "Very Long Column"]
        result = engine.columns(columns, widths=[10, 20, 30])
        assert len(result) > 0


class TestTable:
    """Test table generation."""
    
    def test_simple_table(self):
        """Test creating a simple table."""
        engine = LayoutEngine()
        headers = ["Name", "Age", "City"]
        rows = [
            ["Alice", "30", "NYC"],
            ["Bob", "25", "LA"],
        ]
        result = engine.table(headers, rows)
        # Table should contain all data
        assert "Name" in result
        assert "Alice" in result
        assert "Bob" in result
    
    def test_table_with_borders(self):
        """Test table with borders."""
        engine = LayoutEngine()
        headers = ["A", "B"]
        rows = [["1", "2"]]
        result = engine.table(headers, rows, border=True)
        # Should have border characters
        assert any(c in result for c in ['+', '-', '|', '│', '─'])
    
    def test_empty_table(self):
        """Test creating an empty table."""
        engine = LayoutEngine()
        result = engine.table(["Col"], [])
        assert "Col" in result


class TestPaddingMargin:
    """Test padding and margin classes."""
    
    def test_padding_dataclass(self):
        """Test Padding dataclass."""
        pad = Padding(top=1, right=2, bottom=3, left=4)
        assert pad.top == 1
        assert pad.right == 2
        assert pad.bottom == 3
        assert pad.left == 4
    
    def test_padding_default(self):
        """Test default Padding values."""
        pad = Padding()
        assert pad.top == 0
        assert pad.right == 0
    
    def test_margin_dataclass(self):
        """Test Margin dataclass."""
        margin = Margin(top=1, right=2, bottom=3, left=4)
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
        engine = LayoutEngine()
        text = "The quick brown fox jumps over the lazy dog"
        result = engine.wrap(text, width=10)
        lines = result if isinstance(result, list) else result.split('\n')
        assert len(lines) > 1
    
    def test_wrap_short_text(self):
        """Test wrapping text that fits."""
        engine = LayoutEngine()
        text = "Short"
        result = engine.wrap(text, width=20)
        lines = result if isinstance(result, list) else result.split('\n')
        assert len(lines) == 1
    
    def test_wrap_preserves_words(self):
        """Test that wrapping preserves word boundaries."""
        engine = LayoutEngine()
        text = "Hello World"
        result = engine.wrap(text, width=8)
        lines = result if isinstance(result, list) else result.split('\n')
        # "Hello" should not be split
        assert any("Hello" in line for line in lines)


class TestLayoutEngine:
    """Test LayoutEngine general functionality."""
    
    def test_engine_creation(self):
        """Test creating a LayoutEngine."""
        engine = LayoutEngine()
        assert engine is not None
    
    def test_terminal_width_detection(self):
        """Test terminal width detection."""
        engine = LayoutEngine()
        width = engine.get_terminal_width()
        assert isinstance(width, int)
        assert width > 0
    
    def test_ansi_aware_width(self):
        """Test width calculation ignores ANSI codes."""
        engine = LayoutEngine()
        text_with_ansi = "\033[31mRed\033[0m"
        width = engine.text_width(text_with_ansi)
        # Width should be 3 (just "Red"), not include ANSI codes
        assert width == 3
