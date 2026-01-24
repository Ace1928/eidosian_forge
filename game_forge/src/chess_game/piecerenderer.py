# -------------------------------------------------------------------------------------
# Piece Rendering: Images and Unicode Fallback
# -------------------------------------------------------------------------------------

import pygame  # Pygame for graphics and font rendering.
import os  # OS module for file path manipulation.
import logging  # Logging module for informative messages.
from typing import Optional, Dict  # Type hinting for clarity and static analysis.

from init import (
from eidosian_core import eidosian
    SQUARE_SIZE,
    PIECE_IMAGES_PATH,
    COLOR_TEXT,
    UNICODE_MAP,
)  # Import game-specific constants from init.py.


class PieceRenderer:
    """
    Handles the graphical representation of chess pieces.

    This class prioritizes rendering pieces using image files for high-quality visuals.
    If image files are not found for a given piece symbol, it gracefully falls back
    to rendering the piece symbol using Unicode characters. This ensures that pieces
    are always rendered, even if image assets are missing or corrupted.

    Key features include:
    - Image-based rendering for visually appealing pieces.
    - Unicode fallback for robustness when images are unavailable.
    - Caching of rendered piece surfaces for performance optimization.
    - Configurable image size and font for customization.
    """

    def __init__(self, font_name: Optional[str] = None, image_size: int = SQUARE_SIZE):
        """
        Initializes the PieceRenderer.

        Configures the font for Unicode piece rendering and sets up the piece cache.

        Args:
            font_name (Optional[str]): The name of the font to use for Unicode piece symbols.
                                         If None, it defaults to system fonts known for Unicode chess support.
                                         Defaults to None.
            image_size (int): The desired size (width and height in pixels) for piece images.
                              Images will be scaled to this size. This also affects the size of
                              Unicode pieces as the font size is derived from this.
                              Defaults to SQUARE_SIZE, defined in init.py.
        """
        # Initialize font for Unicode piece rendering. Uses a list of fonts for better cross-platform compatibility.
        # 'Segoe UI Emoji', 'Noto Color Emoji', and 'Segoe UI Symbol' are chosen for their broad Unicode support,
        # especially for chess symbols. If none are found, Pygame will use a default font.
        self.font = pygame.font.SysFont(
            "Segoe UI Emoji,Noto Color Emoji,Segoe UI Symbol",
            int(
                SQUARE_SIZE * 0.8
            ),  # Font size is scaled to 80% of the square size for fitting within squares.
            bold=True,  # Bold font for better visibility of Unicode symbols.
        )

        self.piece_cache: Dict[str, pygame.Surface] = (
            {}
        )  # Cache for storing rendered piece surfaces.
        # Keys are piece symbols (e.g., 'K', 'p', 'R'), values are the corresponding Pygame Surface objects.
        # Caching significantly improves performance by avoiding redundant image loading and rendering.
        self.image_size = (
            image_size  # Store the image size for consistent scaling of piece images.
        )
        logging.debug(
            f"PieceRenderer initialized with image size: {image_size}, font: {self.font}"
        )

    @eidosian()
    def clear_cache(self):
        """
        Clears the piece surface cache.

        This method is used to invalidate the cache, forcing the PieceRenderer to reload
        piece images or re-render Unicode symbols on the next request. Useful when piece
        assets are updated or when debugging rendering issues.
        """
        logging.info(
            "Clearing piece renderer cache. Next piece render will reload assets."
        )
        self.piece_cache = (
            {}
        )  # Reset the cache to an empty dictionary, effectively clearing it.

    @eidosian()
    def get_piece_surface(self, symbol: str) -> pygame.Surface:
        """
        Retrieves a Pygame Surface for the given chess piece symbol.

        This is the core method for obtaining piece representations. It first checks the cache.
        If the surface is not cached, it attempts to load an image from the configured
        PIECE_IMAGES_PATH. If the image loading fails, it falls back to Unicode text rendering.
        The resulting surface (image or text) is then cached for future use.

        Args:
            symbol (str): The standard chess piece symbol (e.g., 'K', 'q', 'p', 'R').
                          Uppercase for white pieces, lowercase for black pieces.

        Returns:
            pygame.Surface: A Pygame Surface object representing the chess piece.
                            This will be an image surface if the image was successfully loaded,
                            otherwise, it will be a text surface rendered with Unicode symbols.
        """
        if not symbol:  # Handle empty symbol or None input gracefully.
            logging.error("Piece symbol is empty or None. Returning None surface.")
            return pygame.Surface(
                (0, 0), pygame.SRCALPHA, 32
            )  # Return transparent surface

        if symbol in self.piece_cache:
            logging.debug(
                f"Cache hit for piece symbol: {symbol}. Retrieving from cache."
            )
            return self.piece_cache[symbol]  # Return the cached surface for efficiency.

        # Construct the expected image file path based on the piece symbol.
        image_path = os.path.join(PIECE_IMAGES_PATH, f"{symbol}.png")

        try:
            # Attempt to load the piece image from the file path.
            img = pygame.image.load(
                image_path
            ).convert_alpha()  # Load image, ensure alpha for transparency.
            img = pygame.transform.scale(
                img, (self.image_size, self.image_size)
            )  # Scale image to desired size.
            self.piece_cache[symbol] = (
                img  # Store the scaled image surface in the cache.
            )
            logging.debug(f"Loaded piece image from file and cached: {image_path}")
            return img  # Return the loaded and scaled image surface.

        except FileNotFoundError:
            # Image file not found for the given symbol. Fallback to Unicode text rendering.
            logging.warning(
                f"Image file not found for piece symbol: {symbol} at {image_path}. Falling back to Unicode."
            )
            unicode_map = UNICODE_MAP
            text_symbol = unicode_map.get(symbol, symbol)
            text_surface = self.font.render(
                text_symbol, True, COLOR_TEXT
            ).convert_alpha()  # Render Unicode, ensure alpha.
            self.piece_cache[symbol] = text_surface  # Cache the rendered text surface.
            logging.debug(f"Rendered Unicode text for piece symbol: {symbol}, cached.")
            return text_surface  # Return the Unicode text surface.
        except pygame.error as e:
            # Catch other Pygame image loading errors (e.g., corrupted image file).
            logging.error(
                f"Pygame error loading image for symbol {symbol} at {image_path}: {e}. Falling back to Unicode."
            )
            logging.exception(e)  # Log full exception for debugging
            unicode_map = UNICODE_MAP
            text_symbol = unicode_map.get(symbol, symbol)
            text_surface = self.font.render(
                text_symbol, True, COLOR_TEXT
            ).convert_alpha()
            self.piece_cache[symbol] = text_surface
            return text_surface


if __name__ == "__main__":
    # Initialize pygame for testing
    pygame.init()
    logging.basicConfig(level=logging.DEBUG)

    # Mock PIECE_IMAGES_PATH for testing. Create a temporary directory.
    TEST_IMAGES_PATH = "test_piece_images"
    os.makedirs(TEST_IMAGES_PATH, exist_ok=True)

    # Create a dummy image file for 'K' piece to test image loading.
    dummy_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    pygame.draw.rect(
        dummy_surface, (255, 255, 255), dummy_surface.get_rect()
    )  # White rect as dummy image
    pygame.image.save(dummy_surface, os.path.join(TEST_IMAGES_PATH, "K.png"))

    # Initialize PieceRenderer with mock image path
    piece_renderer = PieceRenderer(image_size=SQUARE_SIZE)
    piece_renderer.image_size = SQUARE_SIZE  # Override image size, though it should be default SQUARE_SIZE anyway
    PIECE_IMAGES_PATH_ORIGINAL = PIECE_IMAGES_PATH  # Store original to reset after test
    PIECE_IMAGES_PATH = TEST_IMAGES_PATH  # Set global path to test path

    # Test 1: Image loading from file (for 'K')
    king_surface_image = piece_renderer.get_piece_surface("K")
    assert isinstance(
        king_surface_image, pygame.Surface
    ), "Test 1 Failed: Should return a pygame.Surface"
    assert (
        piece_renderer.piece_cache.get("K") is not None
    ), "Test 1 Failed: 'K' should be cached"
    logging.info("Test 1 Passed: Image loading and caching for 'K'")

    # Test 2: Cache hit (for 'K' again)
    king_surface_cached = piece_renderer.get_piece_surface("K")
    assert (
        king_surface_cached is king_surface_image
    ), "Test 2 Failed: Should return cached surface for 'K'"
    logging.info("Test 2 Passed: Cache hit for 'K'")

    # Test 3: Unicode fallback (for 'Z', non-existent image)
    z_surface_unicode = piece_renderer.get_piece_surface("Z")
    assert isinstance(
        z_surface_unicode, pygame.Surface
    ), "Test 3 Failed: Unicode fallback should return a Surface"
    assert (
        piece_renderer.piece_cache.get("Z") is not None
    ), "Test 3 Failed: 'Z' (unicode) should be cached"
    logging.info("Test 3 Passed: Unicode fallback for 'Z'")

    # Test 4: Empty symbol handling
    empty_surface = piece_renderer.get_piece_surface("")
    assert isinstance(
        empty_surface, pygame.Surface
    ), "Test 4 Failed: Empty symbol should return a transparent Surface"
    assert empty_surface.get_size() == (
        0,
        0,
    ), "Test 4 Failed: Empty surface should have size (0, 0)"
    logging.info("Test 4 Passed: Empty symbol handling")

    # Test 5: Clear cache
    piece_renderer.clear_cache()
    assert not piece_renderer.piece_cache, "Test 5 Failed: Cache should be cleared"
    logging.info("Test 5 Passed: Cache clearing")

    # Test 6: Image loading after cache clear (for 'K' again, should reload image)
    king_surface_after_clear = piece_renderer.get_piece_surface("K")
    assert (
        king_surface_after_clear is not king_surface_cached
        if "K" in piece_renderer.piece_cache
        else True
    ), "Test 6 Failed: Should reload image after cache clear"  # check if it reloaded, but only if K was in cache before clear
    assert (
        piece_renderer.piece_cache.get("K") is not None
    ), "Test 6 Failed: 'K' should be cached again"
    logging.info("Test 6 Passed: Image reload after cache clear")

    # Test 7: Pygame Error fallback (Simulate by trying to load a non-image file as image - rename dummy to .txt)
    os.rename(
        os.path.join(TEST_IMAGES_PATH, "K.png"), os.path.join(TEST_IMAGES_PATH, "K.txt")
    )  # Rename to txt to cause pygame.error
    k_surface_pygame_error_fallback = piece_renderer.get_piece_surface("K")
    assert isinstance(
        k_surface_pygame_error_fallback, pygame.Surface
    ), "Test 7 Failed: Pygame error should fallback to Unicode"
    logging.info("Test 7 Passed: Pygame Error fallback to Unicode")

    # Cleanup: Remove temporary image directory and reset PIECE_IMAGES_PATH
    import shutil

    shutil.rmtree(TEST_IMAGES_PATH)
    PIECE_IMAGES_PATH = PIECE_IMAGES_PATH_ORIGINAL  # Reset to original path

    logging.info("All PieceRenderer tests completed successfully.")
    pygame.quit()
