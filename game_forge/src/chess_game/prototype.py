"""
Fun Chess Game Girls - Personality-driven Chess Prototype
=========================================================
A Python-based prototype demonstrating a personality-driven chess game,
integrating a chess engine, Pygame GUI, piece personalities, and a story engine.

To run the game:
  1. Ensure you have Python installed (preferably Python 3.7 or higher).
  2. Install the required dependencies using pip:
     `pip install pygame python-chess requests`
  3. Save this code as a Python file (e.g., fun_chess_game.py).
  4. Run the game from your terminal: `python fun_chess_game.py`

Developed By: Eidos

Features:
  - Chess Engine Integration (python-chess)
  - Pygame-based GUI
  - Piece Personalities
  - Story Engine Stub (LLM Integration)
  - Player Profile Saving/Loading
"""

import pygame  # For the game's GUI
import chess  # For chess logic and board representation
import sys
import os
import json
import requests
import logging
from typing import Dict, List, Optional, Any, Tuple
import threading
import random
from eidosian_core import eidosian

# -------------------------------------------------------------------------------------
# GLOBAL CONFIGURATION SECTION
# -------------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("game.log"),
        logging.StreamHandler(),
    ],
)

BOARD_WIDTH = 8
BOARD_HEIGHT = 8
SQUARE_SIZE = 80

COLOR_LIGHT = (240, 217, 181)
COLOR_DARK = (181, 136, 99)
COLOR_HIGHLIGHT = (186, 202, 43)
COLOR_BORDER = (0, 0, 0)
COLOR_TEXT = (0, 0, 0)
COLOR_BACKGROUND = (200, 200, 200)

SCREEN_WIDTH = BOARD_WIDTH * SQUARE_SIZE
PANEL_HEIGHT = 150
SCREEN_HEIGHT = BOARD_HEIGHT * SQUARE_SIZE + PANEL_HEIGHT

PLAYER_PROFILES_JSON = "player_profiles.json"

DEFAULT_PERSONALITIES = {
    "P": {
        "name": "White Pawn",
        "traits": ["Brave", "Ambitious", "Loyal"],
        "behaviors": "Slowly marches forward, hoping to become a queen one day.",
    },
    "N": {
        "name": "White Knight",
        "traits": ["Chivalrous", "Swift", "Sneaky"],
        "behaviors": "Leaps over allies and enemies, striking unpredictably.",
    },
    "B": {
        "name": "White Bishop",
        "traits": ["Pious", "Diagonal Strategist"],
        "behaviors": "Stays on color, delivering holy retribution at an angle.",
    },
    "R": {
        "name": "White Rook",
        "traits": ["Towering", "Protective"],
        "behaviors": "Moves in straight lines with unbreakable determination.",
    },
    "Q": {
        "name": "White Queen",
        "traits": ["Regal", "Powerful", "Fearless"],
        "behaviors": "Dominates the battlefield with swift and lethal action.",
    },
    "K": {
        "name": "White King",
        "traits": ["Solemn", "Cautious", "Royal"],
        "behaviors": "Moves slowly, must be protected at all costs.",
    },
    "p": {
        "name": "Black Pawn",
        "traits": ["Steadfast", "Stoic", "Hopeful"],
        "behaviors": "Dreams of promotion, defends its territory bravely.",
    },
    "n": {
        "name": "Black Knight",
        "traits": ["Dark Charger", "Trickster"],
        "behaviors": "Appears where least expected, causing chaos among enemies.",
    },
    "b": {
        "name": "Black Bishop",
        "traits": ["Dark Mentor", "Diagonal Mastermind"],
        "behaviors": "Preaches cunning and strikes from the shadows on diagonals.",
    },
    "r": {
        "name": "Black Rook",
        "traits": ["Fortified", "Unyielding"],
        "behaviors": "Forms the walls of defense, sweeping horizontally and vertically.",
    },
    "q": {
        "name": "Black Queen",
        "traits": ["Malicious", "Dominant", "Fearless"],
        "behaviors": "Rules the board with unstoppable might and cunning.",
    },
    "k": {
        "name": "Black King",
        "traits": ["Resilient", "Protective", "Royal"],
        "behaviors": "Keeps watchful oversight, always mindful of check.",
    },
}

DEFAULT_PROFILE = {
    "games_played": 0,
    "wins": 0,
    "losses": 0,
    "preferred_color": "white",
    "unlocked_personalities": ["P", "R", "N"],
}

PIECE_IMAGES_PATH = "piece_images"  # Where you store your local piece images (PNG)
DEFAULT_FONT_SIZE = 48

EMOJI_MAP = {
    "attack": "⚔️",
    "defend": "🛡️",
    "move": "➡️",
    "check": "⚠️",
    "capture": "💥",
    "strategy": "🧠",
    "protect": "🛡️",
    "win": "🏆",
    "lose": "💔",
    "think": "💭",
    "chess": "♟️",
    "king": "👑",
    "queen": "♕",
    "bishop": "⛪",
    "knight": "🐴",
    "rook": "🏰",
    "pawn": "🛡️",
    "danger": "🔥",
    "victory": "🎉",
    "battle": "⚔️",
}

COLOR_MAP = {
    "Brave": (255, 0, 0),
    "Cautious": (0, 0, 255),
    "Loyal": (0, 128, 0),
    "Sneaky": (128, 0, 128),
    "Royal": (255, 215, 0),
}


# -------------------------------------------------------------------------------------
# Text Processing: Formatting, Wrapping, and Coloring
# -------------------------------------------------------------------------------------


class TextProcessor:
    """
    Handles text formatting, processing, and rendering for the story panel in a thread-safe manner.
    """

    def __init__(self, max_width: int, font: pygame.font.Font):
        """
        Args:
            max_width: The maximum width for text wrapping.
            font: A Pygame Font object for rendering text.
        """
        self.max_width = max_width
        self.font = pygame.font.SysFont(
            "Segoe UI Emoji,Noto Color Emoji,Segoe UI Symbol", 18, bold=False
        )
        self.current_lines: List[List[Dict]] = []
        self.scroll_offset = 0
        self.lock = threading.Lock()
        self.cache: Dict[int, pygame.Surface] = {}

    @eidosian()
    def process_text(self, raw_text: str, personalities: Dict[str, Any]) -> None:
        """
        Runs text through emoji insertion, color coding, and line wrapping, then auto-scrolls.
        """
        processed_text = self._add_emojis(raw_text)
        colored_segments = self._apply_color_coding(processed_text, personalities)
        wrapped_lines = self._smart_wrap(colored_segments)

        with self.lock:
            self.current_lines = wrapped_lines
            self._auto_scroll()

    def _add_emojis(self, text: str) -> str:
        """
        Replaces keywords in text with emojis from EMOJI_MAP.
        """
        for word, emoji in EMOJI_MAP.items():
            text = text.replace(f" {word} ", f" {emoji} ")
            text = text.replace(f"{word} ", f"{emoji} ")
            text = text.replace(f" {word}", f" {emoji}")
        return text

    def _apply_color_coding(
        self, text: str, personalities: Dict[str, Any]
    ) -> List[Dict]:
        """
        Applies color to words that match personality traits.
        """
        colored_segments = []
        traits = personalities.get("traits", [])

        for word in text.split():
            segment = {"text": word, "color": (0, 0, 0)}
            for trait in traits:
                if trait.lower() in word.lower():
                    segment["color"] = COLOR_MAP.get(trait, (0, 0, 0))
                    break
            colored_segments.append(segment)
        return colored_segments

    def _smart_wrap(self, segments: List[Dict]) -> List[List[Dict]]:
        """
        Wraps segments into lines not exceeding max_width.
        """
        lines = []
        current_line: List[Dict] = []
        current_width = 0

        for segment in segments:
            words = segment["text"].split()
            for word in words:
                is_emoji = any(e in word for e in EMOJI_MAP.values())
                # Measure word with a preceding space for spacing
                word_width, _ = self.font.size(" " + word if is_emoji else word + " ")

                if current_width + word_width > self.max_width:
                    lines.append(current_line)
                    current_line = [{"text": word, "color": segment["color"]}]
                    current_width = word_width
                else:
                    current_line.append({"text": word, "color": segment["color"]})
                    current_width += word_width

        if current_line:
            lines.append(current_line)
        return lines

    def _auto_scroll(self):
        """
        Keeps the last lines of text visible in the panel.
        """
        line_height = 24
        visible_lines = PANEL_HEIGHT // line_height
        total_lines = len(self.current_lines)
        self.scroll_offset = max(0, (total_lines - visible_lines) * line_height)


# -------------------------------------------------------------------------------------
# Piece Rendering: Images and Unicode Fallback
# -------------------------------------------------------------------------------------


class PieceRenderer:
    """
    Handles the graphical representation of chess pieces, using images or falling back to Unicode.
    """

    def __init__(self, font_name: Optional[str] = None):
        self.font = pygame.font.SysFont(
            "Segoe UI Emoji,Noto Color Emoji,Segoe UI Symbol",
            int(SQUARE_SIZE * 0.8),
            bold=True,
        )
        self.piece_cache: Dict[str, pygame.Surface] = {}
        self.image_size = SQUARE_SIZE

    @eidosian()
    def clear_cache(self):
        """
        Clears the piece surface cache to force re-render or re-load images.
        """
        self.piece_cache = {}

    @eidosian()
    def get_piece_surface(self, symbol: str) -> pygame.Surface:
        """
        Returns a surface for the given piece symbol, using cache, images, or Unicode fallback.
        """
        if symbol in self.piece_cache:
            logging.debug(f"Cache hit for piece symbol: {symbol}")
            return self.piece_cache[symbol]

        # Attempt to load from images
        try:
            image_path = os.path.join(PIECE_IMAGES_PATH, f"{symbol}.png")
            img = pygame.image.load(image_path)
            img = pygame.transform.scale(img, (self.image_size, self.image_size))
            self.piece_cache[symbol] = img
            logging.debug(f"Loaded piece image from file and cached: {image_path}")
            return img
        except FileNotFoundError:
            logging.warning(
                f"Image file not found for piece symbol: {symbol}. Falling back to Unicode."
            )
            unicode_map = {
                "R": "♜",
                "N": "♞",
                "B": "♝",
                "Q": "♛",
                "K": "♚",
                "P": "♟",
                "r": "♖",
                "n": "♘",
                "b": "♗",
                "q": "♕",
                "k": "♔",
                "p": "♙",
            }
            text = unicode_map.get(symbol, symbol)
            text_surface = self.font.render(text, True, COLOR_TEXT)
            self.piece_cache[symbol] = text_surface
            return text_surface


# -------------------------------------------------------------------------------------
# Enhanced Profile Management
# -------------------------------------------------------------------------------------


@eidosian()
def load_player_profiles() -> Dict[str, Dict[str, Any]]:
    """
    Loads player profiles from a JSON file, applying defaults for any missing fields.
    """
    logging.info("Loading player profiles...")
    if not os.path.exists(PLAYER_PROFILES_JSON):
        logging.warning(f"No profile file found: {PLAYER_PROFILES_JSON}.")
        return {}

    try:
        with open(PLAYER_PROFILES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            for player_name, profile in data.items():
                for key, default_value in DEFAULT_PROFILE.items():
                    if key not in profile:
                        profile[key] = default_value
            logging.info("Player profiles loaded and validated.")
            return data
    except Exception as e:
        logging.error(f"Error loading player profiles: {e}")
        return {}


@eidosian()
def save_player_profiles(profiles: Dict[str, Dict[str, Any]]) -> None:
    """
    Saves player profiles to a JSON file using an atomic write strategy.
    """
    logging.info("Saving player profiles...")
    temp_file = PLAYER_PROFILES_JSON + ".tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        os.replace(temp_file, PLAYER_PROFILES_JSON)
        logging.info("Player profiles saved successfully.")
    except Exception as e:
        logging.error(f"Error saving player profiles: {e}")


# -------------------------------------------------------------------------------------
# LLM Story Engine (Minimal Prototype)
# -------------------------------------------------------------------------------------


@eidosian()
def get_ollama_endpoint() -> Optional[str]:
    """
    Checks for an available Ollama endpoint, returning the first responsive one.
    """
    endpoints = ["http://192.168.4.73:11434"]
    for endpoint in endpoints:
        logging.debug(f"Attempting Ollama endpoint: {endpoint}...")
        try:
            r = requests.get(f"{endpoint}/api/tags", timeout=2)
            if r.status_code == 200:
                logging.info(f"Using Ollama endpoint: {endpoint}")
                return endpoint
        except requests.exceptions.RequestException as e:
            logging.debug(f"Connection failed to {endpoint}: {e}")

    logging.warning("No Ollama endpoints available, using mock stories.")
    return None


# Store global base URL once so we don’t repeatedly query
OLLAMA_API_BASE_URL = get_ollama_endpoint()


@eidosian()
def generate_llm_story(
    board: chess.Board,
    personalities: Dict[str, Dict[str, Any]],
    last_move: Optional[chess.Move],
    piece_symbol: str,
    legal_moves: List[str],
) -> str:
    """
    Generates a short story about the current game state using a local LLM (Ollama).
    Falls back to mock text if no endpoint or errors occur.
    """
    logging.info("Generating LLM story...")

    if not last_move:
        return "The game begins with pieces in their starting positions."

    piece_symbol_upper = piece_symbol.upper()
    piece_data = personalities.get(piece_symbol_upper, {})
    traits = ", ".join(piece_data.get("traits", []))

    # LLM prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI actor in an innovative chess game. "
                "Embody the persona of the chess piece you're assigned with these rules:\n"
                "1. Use first-person perspective exclusively\n"
                "2. Incorporate personality traits naturally\n"
                "3. Reference chess-specific terminology\n"
                "4. Express emotions to board situations\n"
                "5. Under 100 words\n"
                "6. Never break character - you ARE the chess piece"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Roleplay as {piece_data.get('name', 'Piece')} ({traits}) "
                f"after moving {last_move.uci()}. Legal moves: {', '.join(legal_moves[:3])}. "
                "Keep response under 100 words."
            ),
        },
    ]

    if not OLLAMA_API_BASE_URL:
        return "🌀 The battlefield clouds my mind..."

    try:
        response = requests.post(
            f"{OLLAMA_API_BASE_URL}/api/chat",
            json={
                "model": "qwen2.5",
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7},
            },
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get(
                "content", "♟️ The piece remains silent..."
            )
        logging.error(f"Story generation failed: {response.text}")
        return "🌀 The battlefield clouds my mind..."
    except requests.exceptions.RequestException as e:
        logging.error(f"LLM request failed: {str(e)}")
        return random.choice(
            [
                "🔌 Connection to the chess mind lost...",
                "⏳ The piece's thoughts are delayed...",
                "♾️ Infinite possibilities overwhelm us...",
            ]
        )


@eidosian()
def warmup_llm_connection(screen: pygame.Surface):
    """
    Simple check to ensure LLM is reachable and loaded. Exits if no connection available.
    """
    font = pygame.font.Font(None, 24)
    screen.fill(COLOR_BACKGROUND)
    text = font.render("Initializing LLM...", True, COLOR_TEXT)
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()

    if not OLLAMA_API_BASE_URL:
        logging.error("No Ollama endpoint available for warmup.")
        sys.exit("Failed to initialize LLM: No Ollama server found.")

    try:
        response = requests.post(
            f"{OLLAMA_API_BASE_URL}/api/chat",
            json={
                "model": "qwen2.5:0.5B",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI actor in an innovative chess game. "
                            "You will provide first-person narratives from the perspective "
                            "of chess pieces, fully embodying their traits."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Confirm your role by describing how you approach a knight's move."
                        ),
                    },
                ],
                "stream": False,
            },
            timeout=10,
        )
        if response.status_code == 200:
            result = response.json()
            logging.info(
                f"LLM Warmup Response: {result.get('message', {}).get('content', '')}"
            )
            return
        logging.error(f"LLM warmup failed: {response.text}")
        sys.exit("Failed to initialize LLM: Warmup failed.")
    except requests.exceptions.RequestException as e:
        logging.error(f"LLM connection failed: {str(e)}")
        sys.exit("Could not connect to Ollama server.")


# -------------------------------------------------------------------------------------
# Main PersonalityChessGame Class
# -------------------------------------------------------------------------------------


class PersonalityChessGame:
    """
    Main orchestrator of the personality-driven chess experience.
    """

    def __init__(
        self, screen: pygame.Surface, personalities: Dict[str, Dict[str, Any]]
    ) -> None:
        logging.info("Initializing PersonalityChessGame...")

        self.screen = screen
        self.board = chess.Board()
        self.game_over = False
        self.selected_square: Optional[int] = None
        self.personalities = personalities
        self.player_profiles = load_player_profiles()

        self.current_player_name = "Player1"
        if self.current_player_name not in self.player_profiles:
            self.player_profiles[self.current_player_name] = DEFAULT_PROFILE
            save_player_profiles(self.player_profiles)

        self.renderer = PieceRenderer()
        self.story_text = ""
        self.story_font = pygame.font.SysFont(
            "Segoe UI Emoji,Noto Color Emoji,Segoe UI Symbol", 18, bold=False
        )
        self.text_panel = pygame.Rect(
            0, BOARD_HEIGHT * SQUARE_SIZE, SCREEN_WIDTH, PANEL_HEIGHT
        )
        self.story_lock = threading.Lock()
        self.text_processor = TextProcessor(SCREEN_WIDTH - 10, self.story_font)
        self.rendered_lines: List[List[Dict]] = []
        self.last_processed_text = ""

        logging.info("PersonalityChessGame initialization complete.")

    @eidosian()
    def draw_board(self) -> None:
        """
        Renders the chessboard squares and pieces, plus the story panel at the bottom.
        """
        board_pixel_width = 8 * SQUARE_SIZE
        board_pixel_height = 8 * SQUARE_SIZE

        board_x = (SCREEN_WIDTH - board_pixel_width) // 2
        board_y = (SCREEN_HEIGHT - board_pixel_height - PANEL_HEIGHT) // 2

        self.text_panel = pygame.Rect(
            0, SCREEN_HEIGHT - PANEL_HEIGHT, SCREEN_WIDTH, PANEL_HEIGHT
        )

        # Draw squares
        for row in range(8):
            for col in range(8):
                rect_x = board_x + col * SQUARE_SIZE
                rect_y = board_y + row * SQUARE_SIZE
                color = COLOR_LIGHT if (row + col) % 2 == 0 else COLOR_DARK
                pygame.draw.rect(
                    self.screen, color, (rect_x, rect_y, SQUARE_SIZE, SQUARE_SIZE)
                )

        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                piece_symbol = piece.symbol()
                piece_surface = self.renderer.get_piece_surface(piece_symbol)
                text_rect = piece_surface.get_rect(
                    center=(
                        board_x + col * SQUARE_SIZE + SQUARE_SIZE / 2,
                        board_y + row * SQUARE_SIZE + SQUARE_SIZE / 2,
                    )
                )
                self.screen.blit(piece_surface, text_rect)

        # Draw text panel background
        self._draw_panel_background()
        self._render_text_lines()

    def _draw_panel_background(self) -> None:
        """
        Draws a semi-transparent gradient background for the story panel.
        """
        panel_rect = pygame.Rect(
            0, SCREEN_HEIGHT - PANEL_HEIGHT, SCREEN_WIDTH, PANEL_HEIGHT
        )
        gradient = pygame.Surface((SCREEN_WIDTH, PANEL_HEIGHT), pygame.SRCALPHA)
        for y in range(PANEL_HEIGHT):
            alpha = int(255 * (0.7 - y / PANEL_HEIGHT * 0.3))
            pygame.draw.line(
                gradient, (255, 255, 255, alpha), (0, y), (SCREEN_WIDTH, y)
            )
        self.screen.blit(gradient, panel_rect)
        pygame.draw.rect(self.screen, COLOR_BORDER, panel_rect, 1)

    def _render_text_lines(self) -> None:
        """
        Renders each processed line of story text, handling emojis distinctly.
        """
        y_start = SCREEN_HEIGHT - PANEL_HEIGHT + 10 - self.text_processor.scroll_offset
        line_height = 24

        for line in self.rendered_lines:
            x = 15
            for segment in line:
                if any(e in segment["text"] for e in EMOJI_MAP.values()):
                    # Render emojis each frame (original design)
                    surface = self.story_font.render(
                        segment["text"], True, segment["color"]
                    )
                    surface = pygame.transform.scale(surface, (20, 20))
                else:
                    if segment["hash"] not in self.text_processor.cache:
                        tmp_surf = self.story_font.render(
                            segment["text"], True, segment["color"]
                        )
                        self.text_processor.cache[segment["hash"]] = tmp_surf
                    surface = self.text_processor.cache[segment["hash"]]
                self.screen.blit(surface, (x, y_start))
                x += surface.get_width() + 5
            y_start += line_height

    @eidosian()
    def highlight_moves(self, moves: List[int]) -> None:
        """
        Highlights legal moves with a semi-transparent overlay.
        """
        for move_sq in moves:
            row = 7 - (move_sq // 8)
            col = move_sq % 8
            highlight_surface = pygame.Surface(
                (SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA
            )
            highlight_surface.fill(
                (COLOR_HIGHLIGHT[0], COLOR_HIGHLIGHT[1], COLOR_HIGHLIGHT[2], 100)
            )
            self.screen.blit(highlight_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    @eidosian()
    def get_square_under_mouse(self) -> Optional[int]:
        """
        Returns the chess square index (0-63) under the mouse cursor, or None if outside.
        """
        mouse_x, mouse_y = pygame.mouse.get_pos()
        col = mouse_x // SQUARE_SIZE
        row = mouse_y // SQUARE_SIZE
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        return (7 - row) * 8 + col

    @eidosian()
    def get_legal_moves_for_square(self, square: int) -> List[int]:
        """
        Returns a list of target squares (0-63) for legal moves from the given square.
        """
        moves = []
        for m in self.board.legal_moves:
            if m.from_square == square:
                moves.append(m.to_square)
        return moves

    @eidosian()
    def handle_resize(self) -> None:
        """
        Adjusts UI elements and squares after a window resize event.
        """
        global SCREEN_WIDTH, SCREEN_HEIGHT, SQUARE_SIZE
        current_w, current_h = self.screen.get_size()
        if (current_w, current_h) == (SCREEN_WIDTH, SCREEN_HEIGHT):
            return

        SCREEN_WIDTH, SCREEN_HEIGHT = current_w, current_h
        SQUARE_SIZE = min(SCREEN_WIDTH // 8, (SCREEN_HEIGHT - PANEL_HEIGHT) // 8)

        self.renderer = PieceRenderer()
        self.text_processor = TextProcessor(SCREEN_WIDTH - 20, self.story_font)
        self.text_panel = pygame.Rect(
            0, SCREEN_HEIGHT - PANEL_HEIGHT, SCREEN_WIDTH, PANEL_HEIGHT
        )
        self.selected_square = None
        logging.info("UI resized and adjusted.")

    @eidosian()
    def main_loop(self) -> None:
        """
        Main game loop for handling events, rendering, and updating.
        """
        logging.info("Starting main game loop.")
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 48)

        clock = pygame.time.Clock()
        last_move: Optional[chess.Move] = None
        running = True

        while running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logging.info("Quit event received.")
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click()
                    elif event.button == 4:
                        self.text_processor.scroll_offset = max(
                            0, self.text_processor.scroll_offset - 20
                        )
                    elif event.button == 5:
                        max_offset = len(self.rendered_lines) * 20 - PANEL_HEIGHT + 30
                        self.text_processor.scroll_offset = min(
                            max_offset, self.text_processor.scroll_offset + 20
                        )
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(
                        event.size, pygame.RESIZABLE | pygame.SCALED
                    )
                    self.handle_resize()

            self.screen.fill(COLOR_BACKGROUND)
            self.draw_board()

            if self.selected_square is not None:
                highlight_squares = self.get_legal_moves_for_square(
                    self.selected_square
                )
                self.highlight_moves(highlight_squares)

            pygame.display.flip()

            if self.board.is_game_over():
                self.game_over = True
                result = self.board.result()
                logging.info(f"Game over. Result: {result}")
                self.update_player_profile_stats(result)
                running = False

            current_move_stack = self.board.move_stack
            if len(current_move_stack) > 0 and current_move_stack[-1] != last_move:
                last_move = current_move_stack[-1]
                moved_piece = self.board.piece_at(last_move.to_square)
                if moved_piece:
                    piece_symbol = moved_piece.symbol()
                    story_moves = [str(m) for m in self.board.legal_moves]
                    story_text = generate_llm_story(
                        self.board,
                        self.personalities,
                        last_move,
                        piece_symbol,
                        story_moves,
                    )
                    self.update_story(story_text)
                    logging.info("New LLM story generated and displayed.")

        pygame.quit()
        logging.info("Pygame quit. Exiting main loop.")
        sys.exit()

    @eidosian()
    def handle_click(self) -> None:
        """
        Manages piece selection and move attempts when the user clicks on the board.
        """
        clicked_square = self.get_square_under_mouse()
        if clicked_square is None:
            return

        if self.selected_square is not None:
            move = chess.Move(self.selected_square, clicked_square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
            else:
                self.selected_square = clicked_square
        else:
            self.selected_square = clicked_square

    @eidosian()
    def update_player_profile_stats(self, result: str) -> None:
        """
        Increments profile stats based on game result and player's preferred color.
        """
        self.player_profiles[self.current_player_name]["games_played"] += 1
        is_white = (
            self.player_profiles[self.current_player_name]["preferred_color"] == "white"
        )

        if (result == "1-0" and is_white) or (result == "0-1" and not is_white):
            self.player_profiles[self.current_player_name]["wins"] += 1
        elif (result == "0-1" and is_white) or (result == "1-0" and not is_white):
            self.player_profiles[self.current_player_name]["losses"] += 1
        elif result == "1/2-1/2":
            current_draws = self.player_profiles[self.current_player_name].get(
                "draws", 0
            )
            self.player_profiles[self.current_player_name]["draws"] = current_draws + 1

        save_player_profiles(self.player_profiles)

    @eidosian()
    def wrap_text(self, text: str, max_width: int) -> List[str]:
        """
        Wraps text into multiple lines to fit within max_width using the story font.
        """
        words = text.split()
        lines = []
        current_line: List[str] = []

        for word in words:
            test_line = " ".join(current_line + [word])
            width, _ = self.story_font.size(test_line)
            if width > max_width:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        lines.append(" ".join(current_line))
        return lines

    @eidosian()
    def update_story(self, new_text: str) -> None:
        """
        Safely updates the story text and triggers asynchronous processing if changed.
        """
        with self.story_lock:
            self.story_text = new_text
        if new_text != self.last_processed_text:
            self.last_processed_text = new_text
            threading.Thread(target=self._async_process_text, daemon=True).start()

    def _async_process_text(self) -> None:
        """
        Background thread for processing and caching story text segments.
        """
        with self.story_lock:
            current_text = self.story_text
        self.text_processor.process_text(current_text, self.personalities)
        self.rendered_lines = self._cache_text_surfaces()

    def _cache_text_surfaces(self) -> List[List[Dict]]:
        """
        Pre-renders text segments and stores them for efficient drawing.
        """
        rendered = []
        with self.text_processor.lock:
            for line in self.text_processor.current_lines:
                rendered_line = []
                for segment in line:
                    text = segment["text"]
                    color = segment["color"]
                    text_hash = hash(f"{text}{color}")
                    width, _ = self.story_font.size(text)
                    rendered_line.append(
                        {
                            "hash": text_hash,
                            "text": text,
                            "color": color,
                            "width": width,
                        }
                    )
                rendered.append(rendered_line)
        return rendered


# -------------------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------------------


@eidosian()
def main() -> None:
    """
    Initializes and runs the Personality Chess Game.
    """
    logging.info("Starting main function...")
    pygame.init()

    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE | pygame.SCALED
    )
    pygame.display.set_caption("Fun Chess Game Girls - Personality Chess Prototype")

    warmup_llm_connection(screen)

    game = PersonalityChessGame(screen, DEFAULT_PERSONALITIES)
    game.handle_resize()  # Ensure correct sizing on start
    game.main_loop()


if __name__ == "__main__":
    main()
