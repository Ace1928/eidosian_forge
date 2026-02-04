"""
⚡ Eidosian Test Suite: Glyph Forge TUI ⚡

Tests for the terminal user interface with comprehensive coverage.
"""

import pytest
from unittest import mock

pytest.importorskip("textual", reason="textual not installed")


class TestTUIImports:
    """Test TUI module imports correctly."""

    def test_import_tui_module(self) -> None:
        """Verify TUI module can be imported."""
        from glyph_forge.ui import tui
        assert hasattr(tui, 'GlyphForgeApp')
        assert hasattr(tui, 'run_tui')

    def test_import_glyph_forge_app(self) -> None:
        """Verify GlyphForgeApp class is available."""
        from glyph_forge.ui.tui import GlyphForgeApp
        assert GlyphForgeApp is not None
        assert GlyphForgeApp.TITLE == "⚡ Glyph Forge ⚡"

    def test_import_custom_widgets(self) -> None:
        """Verify custom widgets are available."""
        from glyph_forge.ui.tui import (
            LabeledInput,
            LabeledSelect,
            LabeledSwitch,
            OutputPanel,
        )
        assert LabeledInput is not None
        assert LabeledSelect is not None
        assert LabeledSwitch is not None
        assert OutputPanel is not None

    def test_import_tab_components(self) -> None:
        """Verify tab components are available."""
        from glyph_forge.ui.tui import (
            BannerTab,
            ImageTab,
            StreamingTab,
            SettingsTab,
            AboutTab,
        )
        assert BannerTab is not None
        assert ImageTab is not None
        assert StreamingTab is not None
        assert SettingsTab is not None
        assert AboutTab is not None


class TestTUIConstants:
    """Test TUI constants and configuration."""

    def test_color_modes(self) -> None:
        """Verify color modes are defined."""
        from glyph_forge.ui.tui import COLOR_MODES
        assert len(COLOR_MODES) >= 4
        mode_values = [m[0] for m in COLOR_MODES]
        assert "none" in mode_values
        assert "truecolor" in mode_values

    def test_dither_algorithms(self) -> None:
        """Verify dither algorithms are defined."""
        from glyph_forge.ui.tui import DITHER_ALGORITHMS
        assert len(DITHER_ALGORITHMS) >= 2
        alg_values = [a[0] for a in DITHER_ALGORITHMS]
        assert "floyd-steinberg" in alg_values

    def test_banner_styles(self) -> None:
        """Verify banner styles are defined."""
        from glyph_forge.ui.tui import BANNER_STYLES
        assert len(BANNER_STYLES) >= 5
        style_values = [s[0] for s in BANNER_STYLES]
        assert "minimal" in style_values
        assert "eidosian" in style_values

    def test_glyph_forge_banner(self) -> None:
        """Verify banner constant exists and is substantial."""
        from glyph_forge.ui.tui import GLYPH_FORGE_BANNER
        # Banner uses Unicode block characters, not ASCII letters
        assert len(GLYPH_FORGE_BANNER) > 100
        assert "█" in GLYPH_FORGE_BANNER  # Contains block characters


class TestGlyphForgeApp:
    """Test GlyphForgeApp class."""

    def test_app_title(self) -> None:
        """Verify app title is Glyph Forge."""
        from glyph_forge.ui.tui import GlyphForgeApp
        assert "Glyph Forge" in GlyphForgeApp.TITLE

    def test_app_bindings(self) -> None:
        """Verify app has required bindings."""
        from glyph_forge.ui.tui import GlyphForgeApp
        binding_keys = [b.key for b in GlyphForgeApp.BINDINGS]
        assert "q" in binding_keys

    def test_app_css_path(self) -> None:
        """Verify CSS path is set."""
        from glyph_forge.ui.tui import GlyphForgeApp
        assert GlyphForgeApp.CSS_PATH == "glyph_forge.css"


class TestCustomWidgets:
    """Test custom widget classes."""

    def test_labeled_input_init(self) -> None:
        """Test LabeledInput initialization."""
        from glyph_forge.ui.tui import LabeledInput
        widget = LabeledInput(
            label="Test Label",
            input_id="test_input",
            placeholder="Enter value",
            value="initial",
        )
        assert widget._label_text == "Test Label"
        assert widget._input_id == "test_input"
        assert widget._placeholder == "Enter value"
        assert widget._value == "initial"

    def test_labeled_select_init(self) -> None:
        """Test LabeledSelect initialization."""
        from glyph_forge.ui.tui import LabeledSelect
        options = [("opt1", "Option 1"), ("opt2", "Option 2")]
        widget = LabeledSelect(
            label="Test Select",
            options=options,
            select_id="test_select",
            value="opt1",
        )
        assert widget._label_text == "Test Select"
        assert widget._select_id == "test_select"
        assert widget._options == options
        assert widget._value == "opt1"

    def test_labeled_switch_init(self) -> None:
        """Test LabeledSwitch initialization."""
        from glyph_forge.ui.tui import LabeledSwitch
        widget = LabeledSwitch(
            label="Test Switch",
            switch_id="test_switch",
            value=True,
        )
        assert widget._label_text == "Test Switch"
        assert widget._switch_id == "test_switch"
        assert widget._value is True


class TestStreamingTab:
    """Test StreamingTab functionality."""

    @pytest.mark.skip(reason="Brittle mock interaction")
    @mock.patch("subprocess.Popen")
    def test_start_stream_video(self, mock_popen) -> None:
        """Test launching video stream."""
        from glyph_forge.ui.tui import StreamingTab
        tab = StreamingTab()
        
        # Mock UI inputs
        with mock.patch.object(tab, '_get_select') as mock_select:
            with mock.patch.object(tab, 'query_one') as mock_query:
                with mock.patch.object(tab, '_get_input', return_value="30"):
                    # Setup mocks
                    def select_side_effect(selector, default):
                        if selector == "#stream_source_type": return "video"
                        if selector == "#stream_resolution": return "720p"
                        if selector == "#stream_mode": return "gradient"
                        if selector == "#stream_color": return "ansi256"
                        return default
                    
                    mock_select.side_effect = select_side_effect
                    
                    mock_input = mock.MagicMock()
                    mock_input.value = "test.mp4"
                    
                    mock_switch = mock.MagicMock()
                    mock_switch.value = True # Audio/Record/Stats enabled
                    
                    def query_side_effect(selector):
                        if selector == "#stream_source": return mock_input
                        return mock_switch
                    
                    mock_query.side_effect = query_side_effect
                    
                    # Call start_stream
                    tab.start_stream()
                    
                    # Verify Popen called
                    mock_popen.assert_called_once()
                    args = mock_popen.call_args.args[0]
                    
                    assert "glyph-forge" in args
                    assert "stream" in args
                    assert "test.mp4" in args
                    assert "--resolution" in args
                    assert "720p" in args
                    assert "--mode" in args
                    assert "gradient" in args

    @pytest.mark.skip(reason="Brittle mock interaction")
    @mock.patch("subprocess.Popen")
    def test_start_stream_webcam(self, mock_popen) -> None:
        """Test launching webcam stream."""
        from glyph_forge.ui.tui import StreamingTab
        tab = StreamingTab()
        
        with mock.patch.object(tab, '_get_select') as mock_select:
            with mock.patch.object(tab, 'query_one') as mock_query:
                with mock.patch.object(tab, '_get_input', return_value="30"):
                    mock_select.side_effect = lambda s, d: "webcam" if s == "#stream_source_type" else d
                    
                    mock_input = mock.MagicMock()
                    mock_input.value = "0"
                    
                    mock_switch = mock.MagicMock()
                    mock_switch.value = True
                    
                    mock_query.side_effect = lambda s: mock_input if s == "#stream_source" else mock_switch
                    
                    tab.start_stream()
                    
                    mock_popen.assert_called_once()
                    args = mock_popen.call_args.args[0]
                    assert "--webcam" in args
                    assert "0" in args


class TestCSSFile:
    """Test CSS file exists and has content."""

    def test_css_file_exists(self) -> None:
        """Verify CSS file exists."""
        from pathlib import Path
        css_path = Path(__file__).parent.parent / "src" / "glyph_forge" / "ui" / "glyph_forge.css"
        assert css_path.exists()

    def test_css_file_has_content(self) -> None:
        """Verify CSS file has substantial content."""
        from pathlib import Path
        css_path = Path(__file__).parent.parent / "src" / "glyph_forge" / "ui" / "glyph_forge.css"
        content = css_path.read_text()
        assert len(content) > 1000  # Should be substantial
        assert "Glyph Forge" in content
        assert "Screen" in content
        assert "Button" in content
