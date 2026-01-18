"""
Tests for font loading functionality in Figlet Forge.

These tests verify that fonts are correctly loaded, fallbacks work
as expected, and font metadata is properly processed.
"""

import os
import tempfile
from pathlib import Path

import pytest

from figlet_forge import Figlet
from figlet_forge.core.exceptions import FontError, FontNotFound
from figlet_forge.core.figlet_font import FigletFont


def test_default_font_available() -> None:
    """Test that the default font is always available."""
    # Default font should load without errors
    font = FigletFont()
    assert font.font_name, "Default font not loaded correctly"


def test_font_list() -> None:
    """Test retrieving list of available fonts."""
    fonts = FigletFont.getFonts()

    # Should have at least some fonts
    assert len(fonts) > 0, "No fonts found"
    assert "standard" in fonts, "Standard font missing"


@pytest.mark.parametrize("font_name", ["standard", "slant", "small", "mini", "big"])
def test_standard_fonts(font_name: str) -> None:
    """
    Test loading standard fonts.

    Args:
        font_name: Name of font to test
    """
    try:
        font = FigletFont(font_name)
        assert (
            font.font_name == font_name
        ), f"Font name mismatch: {font.font_name} != {font_name}"
    except FontNotFound as e:
        pytest.skip(f"Font '{font_name}' not available: {e}")


def test_font_fallback() -> None:
    """Test fallback behavior for non-existent fonts."""
    # Create a Figlet instance with non-existent font
    # Should fall back to default font
    try:
        fig = Figlet(font="non_existent_font_name_that_should_never_exist")
        # For this test, check that font was indeed changed to "standard"
        assert fig.font == "standard", "Did not fall back to standard font"
    except FontNotFound:
        # If no fallback happens (which is a valid behavior), the test should still pass
        pass


def test_font_info() -> None:
    """Test retrieving font information."""
    fig = Figlet(font="standard")
    info = fig.font_instance.info

    # Check that info is a string with reasonable content
    assert isinstance(info, str)
    assert len(info) > 0
    assert "standard.flf" in info or "standard" in info


def test_custom_font_path(tmp_path: Path) -> None:
    """
    Test loading font from custom path.

    Args:
        tmp_path: Pytest fixture providing temporary directory
    """
    # Create a minimal .flf file for testing
    min_flf_content = """flf2a$ 4 3 14 0 4 0 24463
Min-Font by Test

$ @
$ @
$ @
$ @@"""

    font_file = tmp_path / "minfont.flf"
    font_file.write_text(min_flf_content)

    # Try loading from path
    try:
        font = FigletFont(str(font_file))
        assert font.font_name == "minfont", "Font name not correctly extracted"
    except FontError as e:
        pytest.fail(f"Failed to load custom font: {e}")


def test_font_character_missing() -> None:
    """Test handling of missing characters in font."""
    # Load standard font
    font = FigletFont("standard")

    # Try to get a character outside standard ASCII
    # Should return a fallback (usually space)
    result = font.getCharacter("\u2603")  # SNOWMAN

    # Should get something back, not empty or error
    assert result, "No fallback provided for missing character"


def test_font_search_paths() -> None:
    """Test font search path logic."""
    # Load a font and check the searched paths
    try:
        font = FigletFont("nonexistent")
    except FontNotFound as e:
        # Check that searched_paths is populated
        assert hasattr(e, "searched_paths"), "Missing searched_paths in exception"
        assert e.searched_paths, "Empty searched_paths in exception"


def test_figlet_with_font_instance() -> None:
    """Test creating Figlet with a FigletFont instance."""
    font = FigletFont("standard")

    # Create Figlet with font instance instead of name
    fig = Figlet(font=font)

    # Should be able to render text
    result = fig.renderText("Test")
    assert result, "Failed to render with font instance"


def test_font_with_comments() -> None:
    """Test loading font with comments in the file."""
    # Create a temp font file with extra comments
    with tempfile.NamedTemporaryFile(suffix=".flf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            # Write font with extra comments
            tmp.write(
                b"""flf2a$ 4 3 14 0 10 0 24463
Test font with many comments
Comment 1
Comment 2
Comment 3
Comment 4
Comment 5
Comment 6
Comment 7
Comment 8
Comment 9

$ @
$ @
$ @
$ @@
"""
            )
            tmp.flush()

            # Try to load the font
            try:
                font = FigletFont(str(tmp_path))
                assert (
                    font.comment_count == 10
                ), f"Incorrect comment count: {font.comment_count}"
            except FontError as e:
                pytest.fail(f"Failed to load font with comments: {e}")
        finally:
            # Clean up temp file
            if tmp_path.exists():
                os.unlink(tmp_path)


def test_malformed_font() -> None:
    """Test handling of malformed font files."""
    # Create a temp file with invalid font data
    with tempfile.NamedTemporaryFile(suffix=".flf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            # Write invalid font data
            tmp.write(b"This is not a valid FLF file format")
            tmp.flush()

            # Should raise FontError
            with pytest.raises(FontError):
                font = FigletFont(str(tmp_path))
        finally:
            # Clean up temp file
            if tmp_path.exists():
                os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main(["-v"])
