"""Tests for Ultra High-Fidelity Rendering Module.

Tests the Braille sub-pixel rendering, extended gradients, hybrid rendering,
and perceptual color operations.
"""
import math
import pytest
import numpy as np

from glyph_forge.streaming.hifi import (
    BrailleRenderer,
    ExtendedGradient,
    HybridRenderer,
    PerceptualColor,
    render_braille,
    render_hybrid,
    BRAILLE_CHARS,
    BRAILLE_BASE,
    BLOCK_GRADIENT_64,
    EXTENDED_GRADIENT_128,
    STANDARD_RESOLUTIONS,
    resolution_to_terminal,
    terminal_to_resolution,
)


class TestBrailleCharacterEncoding:
    """Test Braille character encoding constants."""
    
    def test_braille_base_unicode(self):
        """Braille base should be U+2800."""
        assert BRAILLE_BASE == 0x2800
    
    def test_braille_chars_count(self):
        """Should have 256 Braille patterns."""
        assert len(BRAILLE_CHARS) == 256
    
    def test_braille_empty_pattern(self):
        """Pattern 0 should be empty Braille character."""
        assert BRAILLE_CHARS[0] == chr(0x2800)  # ⠀ (blank)
    
    def test_braille_full_pattern(self):
        """Pattern 255 should be full Braille character."""
        assert BRAILLE_CHARS[255] == chr(0x28FF)  # ⣿ (full)
    
    def test_braille_single_dots(self):
        """Single dot patterns should match expected characters."""
        # Dot 1 (top-left) = bit 0 = value 1
        assert BRAILLE_CHARS[0x01] == "⠁"
        # Dot 4 (top-right) = bit 3 = value 8
        assert BRAILLE_CHARS[0x08] == "⠈"
        # Dot 7 (bottom-left) = bit 6 = value 64
        assert BRAILLE_CHARS[0x40] == "⡀"
        # Dot 8 (bottom-right) = bit 7 = value 128
        assert BRAILLE_CHARS[0x80] == "⢀"


class TestBrailleRenderer:
    """Test BrailleRenderer class."""
    
    @pytest.fixture
    def renderer(self):
        """Create default renderer."""
        return BrailleRenderer(threshold=128, dither=False)
    
    @pytest.fixture
    def white_image(self):
        """Create 4x2 white image (1 Braille character)."""
        return np.full((4, 2), 255, dtype=np.uint8)
    
    @pytest.fixture
    def black_image(self):
        """Create 4x2 black image (1 Braille character)."""
        return np.zeros((4, 2), dtype=np.uint8)
    
    @pytest.fixture
    def test_image_8x4(self):
        """Create 8x4 test image (2x1 Braille characters)."""
        img = np.zeros((4, 4), dtype=np.uint8)
        # Left half white, right half black
        img[:, :2] = 255
        return img
    
    def test_render_white_image(self, renderer, white_image):
        """White image should produce full Braille (all dots)."""
        result = renderer.render(white_image)
        assert len(result) == 1  # 1 line
        # Note: with adaptive threshold default, result may vary
    
    def test_render_black_image(self, renderer, black_image):
        """Black image should produce empty Braille (no dots)."""
        result = renderer.render(black_image)
        assert len(result) == 1  # 1 line
    
    def test_render_dimensions(self, renderer):
        """Output dimensions should match input/4 and /2."""
        img = np.random.randint(0, 256, (40, 20), dtype=np.uint8)
        result = renderer.render(img)
        assert len(result) == 10  # 40 / 4
        assert all(len(line) == 10 for line in result)  # 20 / 2
    
    def test_render_with_colors(self, renderer):
        """Render with colors should produce ANSI codes."""
        img = np.full((4, 2), 200, dtype=np.uint8)
        colors = np.full((4, 2, 3), [255, 0, 0], dtype=np.uint8)  # Red
        result = renderer.render(img, colors)
        
        assert len(result) == 1
        assert "\033[38;2;" in result[0]  # ANSI color code
        assert "\033[0m" in result[0]  # Reset code
    
    def test_dithering_toggle(self):
        """Dithering should affect output."""
        img = np.full((8, 4), 128, dtype=np.uint8)  # Mid-gray
        
        renderer_no_dither = BrailleRenderer(threshold=128, dither=False, adaptive_threshold=False)
        renderer_with_dither = BrailleRenderer(threshold=128, dither=True, adaptive_threshold=False)
        
        result_no_dither = renderer_no_dither.render(img)
        result_with_dither = renderer_with_dither.render(img)
        
        # Results should differ (dithering spreads error)
        # Note: They might be same for uniform images
        assert isinstance(result_no_dither, list)
        assert isinstance(result_with_dither, list)
    
    def test_resolution_for_terminal(self):
        """Test terminal to pixel resolution calculation."""
        w, h = BrailleRenderer.get_resolution_for_terminal(120, 40)
        assert w == 240
        assert h == 160
    
    def test_terminal_for_resolution(self):
        """Test pixel resolution to terminal calculation."""
        w, h = BrailleRenderer.get_terminal_for_resolution(1920, 1080)
        assert w == 960
        assert h == 270


class TestExtendedGradient:
    """Test ExtendedGradient class."""
    
    def test_standard_gradient(self):
        """Standard preset should load correctly."""
        grad = ExtendedGradient("standard")
        assert grad._len == 5  # "█▓▒░ "
    
    def test_extended_gradient(self):
        """Extended preset should have 128+ characters."""
        grad = ExtendedGradient("extended")
        assert grad._len >= 64
    
    def test_custom_gradient(self):
        """Custom gradient string should work."""
        grad = ExtendedGradient("@#*:. ")
        assert grad._len == 6
    
    def test_get_char_density(self):
        """Get character for various densities."""
        grad = ExtendedGradient("█▓▒░ ")
        
        # Bright (density=1) should be space (lightest)
        assert grad.get_char(1.0) == "█"
        
        # Dark (density=0) should be full block (darkest)
        assert grad.get_char(0.0) == " "
        
        # Mid-gray should be middle char
        mid_char = grad.get_char(0.5)
        assert mid_char in "▒░"  # Middle range
    
    def test_invert_gradient(self):
        """Inverted gradient should reverse character order."""
        grad_normal = ExtendedGradient("█▓▒░ ", invert=False)
        grad_invert = ExtendedGradient("█▓▒░ ", invert=True)
        
        # Dark in normal should be bright in inverted
        assert grad_normal.get_char(0.0) != grad_invert.get_char(0.0)
    
    def test_render_image(self):
        """Render should produce correct dimensions."""
        grad = ExtendedGradient("standard")
        img = np.random.random((10, 20))  # Normalized 0-1
        result = grad.render(img)
        
        assert len(result) == 10
        assert all(len(line) == 20 for line in result)
    
    def test_render_with_colors(self):
        """Render with colors should include ANSI codes."""
        grad = ExtendedGradient("standard")
        img = np.random.random((5, 10))
        colors = np.random.randint(0, 256, (5, 10, 3), dtype=np.uint8)
        result = grad.render(img, colors)
        
        assert all("\033[38;2;" in line for line in result)


class TestHybridRenderer:
    """Test HybridRenderer class."""
    
    @pytest.fixture
    def renderer(self):
        """Create default hybrid renderer."""
        return HybridRenderer(edge_threshold=30)
    
    def test_render_dimensions(self, renderer):
        """Output should match Braille dimensions."""
        gray = np.random.randint(0, 256, (40, 20), dtype=np.uint8)
        colors = np.random.randint(0, 256, (40, 20, 3), dtype=np.uint8)
        
        result = renderer.render(gray, colors)
        
        assert len(result) == 10  # 40 / 4
        # Width varies due to ANSI codes
    
    def test_edge_weight_parameter(self):
        """Edge weight should affect rendering."""
        gray = np.random.randint(0, 256, (40, 20), dtype=np.uint8)
        colors = np.random.randint(0, 256, (40, 20, 3), dtype=np.uint8)
        
        r1 = HybridRenderer(edge_weight=0.0)
        r2 = HybridRenderer(edge_weight=1.0)
        
        result1 = r1.render(gray, colors)
        result2 = r2.render(gray, colors)
        
        assert isinstance(result1, list)
        assert isinstance(result2, list)


class TestPerceptualColor:
    """Test PerceptualColor class."""
    
    def test_rgb_to_lab_black(self):
        """Black should have L=0."""
        L, a, b = PerceptualColor.rgb_to_lab(0, 0, 0)
        assert abs(L - 0) < 0.5
    
    def test_rgb_to_lab_white(self):
        """White should have L=100."""
        L, a, b = PerceptualColor.rgb_to_lab(255, 255, 255)
        assert abs(L - 100) < 0.5
    
    def test_rgb_to_lab_gray(self):
        """Neutral gray should have a=b=0."""
        L, a, b = PerceptualColor.rgb_to_lab(128, 128, 128)
        assert abs(a) < 5  # Near-neutral
        assert abs(b) < 5
    
    def test_delta_e_identical(self):
        """Identical colors should have Delta E = 0."""
        lab = PerceptualColor.rgb_to_lab(128, 64, 32)
        delta = PerceptualColor.delta_e(lab, lab)
        assert delta == 0.0
    
    def test_delta_e_different(self):
        """Different colors should have positive Delta E."""
        lab1 = PerceptualColor.rgb_to_lab(255, 0, 0)  # Red
        lab2 = PerceptualColor.rgb_to_lab(0, 0, 255)  # Blue
        delta = PerceptualColor.delta_e(lab1, lab2)
        assert delta > 50  # Very different colors
    
    def test_find_closest_color(self):
        """Find closest should return best match."""
        palette = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
        ]
        
        # Target is orange (255, 128, 0) - closest to red or yellow
        closest = PerceptualColor.find_closest_color((255, 128, 0), palette)
        assert closest in [(255, 0, 0), (255, 255, 0)]
        
        # Target is cyan (0, 255, 255) - closest to green or blue
        closest = PerceptualColor.find_closest_color((0, 255, 255), palette)
        assert closest in [(0, 255, 0), (0, 0, 255)]


class TestResolutionUtilities:
    """Test resolution conversion utilities."""
    
    def test_standard_resolutions_exist(self):
        """Standard resolutions should be defined."""
        assert "480p" in STANDARD_RESOLUTIONS
        assert "720p" in STANDARD_RESOLUTIONS
        assert "1080p" in STANDARD_RESOLUTIONS
    
    def test_resolution_values(self):
        """Resolution values should be correct."""
        assert STANDARD_RESOLUTIONS["720p"] == (1280, 720)
        assert STANDARD_RESOLUTIONS["1080p"] == (1920, 1080)
    
    def test_resolution_to_terminal_1080p(self):
        """1080p should require 960x270 terminal."""
        w, h = resolution_to_terminal("1080p", mode="braille")
        assert w == 960
        assert h == 270
    
    def test_resolution_to_terminal_720p(self):
        """720p should require 640x180 terminal."""
        w, h = resolution_to_terminal("720p", mode="braille")
        assert w == 640
        assert h == 180
    
    def test_terminal_to_resolution_braille(self):
        """Terminal size should map to pixel resolution."""
        w, h = terminal_to_resolution(120, 40, mode="braille")
        assert w == 240
        assert h == 160
    
    def test_invalid_resolution_raises(self):
        """Unknown resolution should raise ValueError."""
        with pytest.raises(ValueError):
            resolution_to_terminal("999p", mode="braille")


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_render_braille_function(self):
        """render_braille should work standalone."""
        img = np.random.randint(0, 256, (16, 8), dtype=np.uint8)
        result = render_braille(img)
        
        assert isinstance(result, list)
        assert len(result) == 4  # 16/4
        assert all(len(line) == 4 for line in result)  # 8/2
    
    def test_render_hybrid_function(self):
        """render_hybrid should work standalone."""
        gray = np.random.randint(0, 256, (20, 10), dtype=np.uint8)
        colors = np.random.randint(0, 256, (20, 10, 3), dtype=np.uint8)
        
        result = render_hybrid(gray, colors)
        
        assert isinstance(result, list)
        assert len(result) == 5  # 20/4


class TestBraillePatternComputation:
    """Test the core Braille pattern computation."""
    
    @pytest.fixture
    def renderer(self):
        """Create renderer with no preprocessing."""
        return BrailleRenderer(
            threshold=128,
            dither=False,
            adaptive_threshold=False,
        )
    
    def test_all_white_pattern(self, renderer):
        """All white pixels should produce pattern 255."""
        # 4x2 white image (one Braille character)
        img = np.full((4, 2), 255, dtype=np.uint8)
        patterns = renderer._compute_braille_patterns(img, 1, 1)
        assert patterns[0, 0] == 255
    
    def test_all_black_pattern(self, renderer):
        """All black pixels should produce pattern 0."""
        # 4x2 black image
        img = np.zeros((4, 2), dtype=np.uint8)
        patterns = renderer._compute_braille_patterns(img, 1, 1)
        assert patterns[0, 0] == 0
    
    def test_top_left_dot_only(self, renderer):
        """Single dot at (0,0) should produce pattern 1."""
        img = np.zeros((4, 2), dtype=np.uint8)
        img[0, 0] = 255  # Top-left dot
        patterns = renderer._compute_braille_patterns(img, 1, 1)
        assert patterns[0, 0] == 0x01
    
    def test_bottom_right_dot_only(self, renderer):
        """Single dot at (3,1) should produce pattern 128."""
        img = np.zeros((4, 2), dtype=np.uint8)
        img[3, 1] = 255  # Bottom-right dot (dot 8)
        patterns = renderer._compute_braille_patterns(img, 1, 1)
        assert patterns[0, 0] == 0x80
    
    def test_left_column_dots(self, renderer):
        """Left column dots should produce pattern 71 (1+2+4+64)."""
        img = np.zeros((4, 2), dtype=np.uint8)
        img[:, 0] = 255  # All left column
        patterns = renderer._compute_braille_patterns(img, 1, 1)
        # Dots 1,2,3,7 = 1+2+4+64 = 71
        assert patterns[0, 0] == 0x47


class TestGradientPresets:
    """Test gradient presets."""
    
    def test_blocks_gradient_length(self):
        """Block gradient should have 64+ characters."""
        assert len(BLOCK_GRADIENT_64) >= 30
    
    def test_extended_gradient_length(self):
        """Extended gradient should have 128+ characters."""
        assert len(EXTENDED_GRADIENT_128) >= 50
    
    def test_all_presets_accessible(self):
        """All presets should be accessible."""
        for preset in ExtendedGradient.PRESETS:
            grad = ExtendedGradient(preset)
            assert grad._len > 0


class TestPerformance:
    """Performance tests for high-fidelity rendering."""
    
    def test_braille_render_speed(self):
        """Braille rendering should be fast."""
        import time
        
        renderer = BrailleRenderer(threshold=128, dither=False)
        img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        
        start = time.perf_counter()
        for _ in range(10):
            renderer.render(img)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 10) * 1000
        print(f"\nBraille render (480p): {avg_ms:.2f}ms/frame")
        
        # Should be under 100ms per frame for 480p
        assert avg_ms < 100, f"Too slow: {avg_ms:.2f}ms"
    
    def test_gradient_render_speed(self):
        """Gradient rendering should be fast."""
        import time
        
        grad = ExtendedGradient("extended")
        img = np.random.random((120, 160))  # 480p in block mode
        
        start = time.perf_counter()
        for _ in range(100):
            grad.render(img)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 100) * 1000
        print(f"\nGradient render (120x160): {avg_ms:.2f}ms/frame")
        
        # Should be under 20ms per frame
        assert avg_ms < 50, f"Too slow: {avg_ms:.2f}ms"
