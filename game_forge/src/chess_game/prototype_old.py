"""
Fun Chess Game Girls - Personality-driven Chess Prototype
=========================================================
Copy and paste this entire code into a single Python file (e.g. `fun_chess_game.py`).
Then run: `python fun_chess_game.py`

Dependencies:
  - pygame
  - python-chess
  - requests (for optional local LLM call)

This prototype demonstrates:
  1. A chess engine for valid moves (using python-chess).
  2. A Pygame-based GUI.
  3. Custom personality data for pieces.
  4. A story engine stub that can interface with a local LLM (ollama, etc.).
  5. Basic player profile saving/loading (JSON).

By: Eidos
"""

import pygame  # For GUI
import chess  # The python-chess engine for move logic
import sys  # System-level operations
import os  # For file/directory checks
import json  # For reading/writing JSON files (player profiles, personalities)
import requests  # For local LLM calls (if desired/needed)
import logging  # For detailed logging
from typing import Dict, List, Optional, Any, Generator
import threading
from eidosian_core import eidosian

# -------------------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# -------------------------------------------------------------------------------------

# Set up logging for detailed output
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the dimensions of the chessboard (8x8).
BOARD_WIDTH = 8
BOARD_HEIGHT = 8
logging.debug(f"Board dimensions set to {BOARD_WIDTH}x{BOARD_HEIGHT}")

# Define how large each square on the board should be in the Pygame window.
SQUARE_SIZE = 80
logging.debug(f"Square size set to {SQUARE_SIZE} pixels")

# Define colors (R, G, B) for squares, highlights, text, etc.
COLOR_LIGHT = (240, 217, 181)  # Light squares
COLOR_DARK = (181, 136, 99)  # Dark squares
COLOR_HIGHLIGHT = (186, 202, 43)  # Highlight color for possible moves
COLOR_BORDER = (0, 0, 0)  # Black border for edges
COLOR_TEXT = (0, 0, 0)  # Black text for piece letters
COLOR_BACKGROUND = (200, 200, 200)  # Background color
logging.debug(
    f"Colors defined: LIGHT={COLOR_LIGHT}, DARK={COLOR_DARK}, HIGHLIGHT={COLOR_HIGHLIGHT}, BORDER={COLOR_BORDER}, TEXT={COLOR_TEXT}, BACKGROUND={COLOR_BACKGROUND}"
)


# Screen width and height (based on board size).
SCREEN_WIDTH = BOARD_WIDTH * SQUARE_SIZE
SCREEN_HEIGHT = BOARD_HEIGHT * SQUARE_SIZE + 150
logging.debug(f"Screen dimensions set to {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# Path to where we might store player profiles in JSON format.
PLAYER_PROFILES_JSON = "player_profiles.json"
logging.debug(f"Player profiles will be stored in: {PLAYER_PROFILES_JSON}")

# If you want to store chess piece personalities externally, you can store them in a JSON.
# For now, we define them inline for the prototype. You can expand these or load from a file.
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
logging.debug(f"Default personalities loaded: {DEFAULT_PERSONALITIES}")

# Default structure for new player profiles
DEFAULT_PROFILE = {
    "games_played": 0,
    "wins": 0,
    "losses": 0,
    "preferred_color": "white",
    "unlocked_personalities": ["P", "R", "N"],  # Example progression system
}
logging.debug(f"Default player profile structure: {DEFAULT_PROFILE}")

# Add this near top with other constants
PIECE_IMAGES_PATH = "piece_images"
DEFAULT_FONT_SIZE = 48

# Add these constants near screen dimensions
PANEL_HEIGHT = 150
SCREEN_HEIGHT = BOARD_HEIGHT * SQUARE_SIZE + PANEL_HEIGHT

# Add near top with other constants
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
    "Brave": (255, 0, 0),  # Red
    "Cautious": (0, 0, 255),  # Blue
    "Loyal": (0, 128, 0),  # Green
    "Sneaky": (128, 0, 128),  # Purple
    "Royal": (255, 215, 0),  # Gold
}


class TextProcessor:
    """Handles text formatting and processing in threads"""

    def __init__(self, max_width: int, font: pygame.font.Font):
        self.max_width = max_width
        self.font = font
        self.current_lines: list[list[dict]] = []
        self.scroll_offset = 0
        self.lock = threading.Lock()
        self.cache: dict[int, pygame.Surface] = {}

    @eidosian()
    def process_text(self, raw_text: str, personalities: Dict[str, Any]) -> None:
        """Thread-safe text processing with emojis and colors"""
        processed = self._add_emojis(raw_text)
        colored = self._apply_color_coding(processed, personalities)
        wrapped = self._smart_wrap(colored)

        with self.lock:
            self.current_lines = wrapped
            self._auto_scroll()

    def _add_emojis(self, text: str) -> str:
        """Replace keywords with emojis"""
        for word, emoji in EMOJI_MAP.items():
            text = text.replace(f" {word} ", f" {emoji} ")
            text = text.replace(f"{word} ", f"{emoji} ")
            text = text.replace(f" {word}", f" {emoji}")
        return text

    def _apply_color_coding(
        self, text: str, personalities: Dict[str, Any]
    ) -> List[Dict]:
        """Apply color coding to personality-related terms"""
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
        """Context-aware wrapping that keeps color segments together"""
        lines = []
        current_line: list[dict] = []
        current_width = 0

        for segment in segments:
            word = segment["text"]
            word_width, _ = self.font.size(word + " ")

            if current_width + word_width > self.max_width:
                lines.append(current_line)
                current_line = [segment]
                current_width = word_width
            else:
                current_line.append(segment)
                current_width += word_width

        if current_line:
            lines.append(current_line)

        return lines

    def _auto_scroll(self):
        """Adjust scroll offset to keep new content visible"""
        total_height = len(self.current_lines) * 20
        if total_height > PANEL_HEIGHT - 10:
            self.scroll_offset = total_height - (PANEL_HEIGHT - 30)


class PieceRenderer:
    """Handles graphical representation of chess pieces"""

    def __init__(self, font_name: str | None = None, image_size: int = SQUARE_SIZE):
        self.font = pygame.font.SysFont(font_name, DEFAULT_FONT_SIZE)
        self.image_size = image_size
        self.piece_cache: dict[str, pygame.Surface] = {}  # Cache loaded images

    @eidosian()
    def get_piece_surface(self, symbol: str) -> pygame.Surface:
        """Get graphical representation for a piece symbol"""
        if symbol in self.piece_cache:
            return self.piece_cache[symbol]

        # Try loading from image file
        try:
            img = pygame.image.load(f"{PIECE_IMAGES_PATH}/{symbol}.png")
            img = pygame.transform.scale(img, (self.image_size, self.image_size))
            self.piece_cache[symbol] = img
            return img
        except FileNotFoundError:
            # Fallback to text rendering
            text_surface = self.font.render(symbol, True, COLOR_TEXT)
            self.piece_cache[symbol] = text_surface
            return text_surface


# -------------------------------------------------------------------------------------
# ENHANCED PROFILE MANAGEMENT
# -------------------------------------------------------------------------------------


@eidosian()
def load_player_profiles() -> Dict[str, Dict[str, Any]]:
    """
    Load player profiles from a JSON file, with enhanced validation and default structure.

    Returns:
        A dictionary containing player profiles, or an empty dictionary if loading fails.
    """
    logging.info("Loading player profiles...")
    # Check if the file exists.
    if not os.path.exists(PLAYER_PROFILES_JSON):
        logging.warning(
            f"Player profiles file not found: {PLAYER_PROFILES_JSON}. Returning empty profiles."
        )
        return {}

    # Try reading and parsing JSON from the file.
    try:
        with open(PLAYER_PROFILES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            logging.debug(
                f"Successfully loaded player profiles from {PLAYER_PROFILES_JSON}: {data}"
            )

            # Validate and update existing profiles with any missing fields
            for player_name, profile in data.items():
                for key, default_value in DEFAULT_PROFILE.items():
                    if key not in profile:
                        profile[key] = default_value
                        logging.debug(
                            f"Player {player_name}: Added missing key '{key}' with default value '{default_value}'."
                        )
            logging.info("Player profiles loaded and validated successfully.")
            return data
    except Exception as e:
        logging.error(f"Error loading player profiles from {PLAYER_PROFILES_JSON}: {e}")
        return {}


@eidosian()
def save_player_profiles(profiles: Dict[str, Dict[str, Any]]) -> None:
    """
    Save player profiles to a JSON file using an atomic write pattern to prevent corruption.

    Args:
        profiles: A dictionary containing player profiles to save.
    """
    logging.info("Saving player profiles...")
    try:
        # Write to temporary file first
        temp_file = PLAYER_PROFILES_JSON + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
            logging.debug(
                f"Successfully wrote player profiles to temporary file: {temp_file}"
            )

        # Replace original file
        os.replace(temp_file, PLAYER_PROFILES_JSON)
        logging.debug(f"Successfully replaced {PLAYER_PROFILES_JSON} with {temp_file}")
        logging.info("Player profiles saved successfully.")
    except Exception as e:
        logging.error(f"Error saving player profiles to {PLAYER_PROFILES_JSON}: {e}")


# -------------------------------------------------------------------------------------
# LLM STORY ENGINE (STUB / DEMO)
# -------------------------------------------------------------------------------------


@eidosian()
def get_ollama_endpoint():
    """Check available Ollama endpoints with timeout"""
    endpoints = ["http://192.168.4.73:11434"]

    for endpoint in endpoints:
        try:
            response = requests.get(f"{endpoint}/api/tags", timeout=2)
            if response.status_code == 200:
                logging.info(f"Using Ollama endpoint: {endpoint}")
                return endpoint
        except Exception as e:
            logging.debug(f"Connection failed to {endpoint}: {e}")

    logging.warning("No Ollama endpoints available, using mock stories")
    return None


OLLAMA_API_BASE_URL = get_ollama_endpoint()


@eidosian()
def generate_llm_story(
    board: chess.Board,
    personalities: Dict[str, Dict[str, Any]],
    last_move: Optional[chess.Move],
    piece_symbol: str,
    legal_moves: List[str],
) -> Generator[str, None, None]:
    """
    This function interfaces with a local LLM (Ollama) to generate a story or narrative
    about the current board situation.

    Args:
        board: The current state of the chess board.
        personalities: A dictionary containing personality data for each piece.
        last_move: The last move made in the game, or None if no moves have been made.
        piece_symbol: The symbol of the piece that moved.
        legal_moves: A list of suggested moves for the current turn.

    Returns:
        A generator that yields story chunks.
    """
    logging.info("Generating LLM story...")

    # Get personality data for moved piece
    piece_data = personalities.get(piece_symbol.upper(), {})
    traits = ", ".join(piece_data.get("traits", []))

    # Build proper chat messages array
    messages = [
        {
            "role": "system",
            "content": "You are a chess piece in an ongoing game. Provide short, vivid first-person narratives.",
        },
        {
            "role": "user",
            "content": f"Roleplay as {piece_data.get('name', 'Piece')} ({traits}) after moving {last_move.uci()}. "
            f"Legal moves: {', '.join(legal_moves[:3])}. Keep response under 100 words.",
        },
    ]

    try:
        # Use /api/chat endpoint with streaming disabled
        response = requests.post(
            f"{get_ollama_endpoint()}/api/chat",
            json={
                "model": "qwen2.5",
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7},
            },
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("done", False):
                yield result["message"]["content"]
                return

        logging.error(f"Story generation failed: {response.text}")
        yield "My thoughts are unclear..."

    except Exception as e:
        logging.error(f"LLM request failed: {str(e)}")
        yield "The battlefield clouds my mind..."


@eidosian()
def warmup_llm_connection():
    """Ensure LLM connection is working before game starts"""
    logging.info("Initializing LLM connection...")
    endpoint = get_ollama_endpoint()

    try:
        # Use /api/chat instead of /api/generate for proper chat completion
        response = requests.post(
            f"{endpoint}/api/chat",
            json={
                "model": "qwen2.5:0.5B",
                "messages": [{"role": "user", "content": "Test connection"}],
                "stream": False,
            },
            timeout=10,
        )

        # Check for successful response with done=true
        if response.status_code == 200:
            result = response.json()
            if result.get("done", False):
                logging.info("LLM warmup successful")
                return

        logging.error(f"LLM warmup failed: {response.text}")
        sys.exit("Failed to initialize LLM connection")

    except Exception as e:
        logging.error(f"LLM connection failed: {str(e)}")
        sys.exit("Could not connect to Ollama server")


# -------------------------------------------------------------------------------------
# CHESS GAME CLASS
# -------------------------------------------------------------------------------------


class PersonalityChessGame:
    """
    The main chess game class integrating:
      - python-chess board
      - personalities
      - Pygame-based UI
      - Story generation
      - Player profiles
    """

    def __init__(
        self, screen: pygame.Surface, personalities: Dict[str, Dict[str, Any]]
    ):
        """
        Initializes the chess game.

        Args:
            screen: The Pygame screen to draw on.
            personalities: A dictionary of piece personalities.
        """
        logging.info("Initializing PersonalityChessGame...")
        # Store reference to the pygame screen so we can draw on it
        self.screen = screen
        logging.debug("Pygame screen reference stored.")

        # Create a python-chess Board object
        self.board = chess.Board()
        logging.debug("Chess board initialized.")

        # A variable to track if the game is finished
        self.game_over = False
        logging.debug("Game over flag initialized to False.")

        # Keep track of the selected square (if any) for movement
        self.selected_square: Optional[int] = None
        logging.debug("Selected square initialized to None.")

        # The dictionary of piece personalities
        self.personalities = personalities
        logging.debug("Personalities dictionary stored.")

        # We can store or load player profiles
        self.player_profiles = load_player_profiles()
        logging.debug("Player profiles loaded.")

        # Current player (can expand to have separate player objects)
        # For demonstration, just store name of the current user
        self.current_player_name = "Player1"
        logging.debug(f"Current player name set to: {self.current_player_name}")

        # If the player doesn't exist in profiles, create an empty record
        if self.current_player_name not in self.player_profiles:
            self.player_profiles[self.current_player_name] = DEFAULT_PROFILE
            save_player_profiles(self.player_profiles)
            logging.info(f"New player profile created for: {self.current_player_name}")
        logging.info("PersonalityChessGame initialized successfully.")

        self.renderer = PieceRenderer()  # Add this line
        self.story_text = ""
        self.story_font = pygame.font.SysFont("Arial", 18)
        self.text_panel = pygame.Rect(
            0, BOARD_HEIGHT * SQUARE_SIZE, SCREEN_WIDTH, PANEL_HEIGHT
        )
        self.story_lock = threading.Lock()
        self.text_processor = TextProcessor(SCREEN_WIDTH - 10, self.story_font)
        self.rendered_lines: list[list[dict]] = []
        self.last_processed_text = ""

    @eidosian()
    def draw_board(self) -> None:
        """
        Draw the chess board squares and the pieces on the screen.
        """
        logging.debug("Drawing chess board...")
        # Loop over each row and column
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                # Determine if square should be light or dark
                if (row + col) % 2 == 0:
                    color = COLOR_LIGHT
                else:
                    color = COLOR_DARK

                # Position (top-left corner) for the square in pixels
                rect_x = col * SQUARE_SIZE
                rect_y = row * SQUARE_SIZE

                # Draw the square
                pygame.draw.rect(
                    self.screen, color, (rect_x, rect_y, SQUARE_SIZE, SQUARE_SIZE)
                )
        logging.debug("Chess board squares drawn.")

        # After drawing squares, we draw pieces on top
        # We can get the piece on each square from self.board
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                # The 'square' is an integer 0..63, let's convert it to row/col
                # python-chess uses 0 for A8, 1 for B8,... 8 for A7, etc.
                # row from top = square // 8, col from left = square % 8
                row = 7 - (square // 8)
                col = square % 8

                # Convert piece symbol to a text representation
                piece_symbol = piece.symbol()  # e.g. 'R' or 'p'
                piece_surface = self.renderer.get_piece_surface(piece_symbol)
                text_rect = piece_surface.get_rect(
                    center=(
                        col * SQUARE_SIZE + SQUARE_SIZE / 2,
                        row * SQUARE_SIZE + SQUARE_SIZE / 2,
                    )
                )
                self.screen.blit(piece_surface, text_rect)
        logging.debug("Chess pieces drawn.")

        # Draw story panel with gradient background
        self._draw_panel_background()
        self._render_text_lines()

    def _draw_panel_background(self):
        """Draw gradient background for text panel"""
        panel_rect = pygame.Rect(
            0, SCREEN_HEIGHT - PANEL_HEIGHT, SCREEN_WIDTH, PANEL_HEIGHT
        )
        gradient = pygame.Surface((SCREEN_WIDTH, PANEL_HEIGHT), pygame.SRCALPHA)
        for y in range(PANEL_HEIGHT):
            alpha = int(255 * (0.7 - y / PANEL_HEIGHT * 0.3))  # More visible gradient
            pygame.draw.line(
                gradient, (255, 255, 255, alpha), (0, y), (SCREEN_WIDTH, y)
            )
        self.screen.blit(gradient, panel_rect)
        pygame.draw.rect(self.screen, COLOR_BORDER, panel_rect, 1)

    def _render_text_lines(self):
        """Render cached text lines with scroll offset"""
        y_start = BOARD_HEIGHT * SQUARE_SIZE + 5 - self.text_processor.scroll_offset
        for line in self.rendered_lines:
            x = 5
            for segment in line:
                if segment["hash"] not in self.text_processor.cache:
                    surface = self.story_font.render(
                        segment["text"], True, segment["color"]
                    )
                    self.text_processor.cache[segment["hash"]] = surface
                self.screen.blit(
                    self.text_processor.cache[segment["hash"]], (x, y_start)
                )
                x += segment["width"]
            y_start += 20

    @eidosian()
    def highlight_moves(self, moves: List[int]) -> None:
        """
        Highlight possible moves (list of squares) on the board.

        Args:
            moves: A list of square indices to highlight.
        """
        logging.debug(f"Highlighting moves: {moves}")
        for move_sq in moves:
            # move_sq is an integer from python-chess representing the board square
            row = 7 - (move_sq // 8)
            col = move_sq % 8

            rect_x = col * SQUARE_SIZE
            rect_y = row * SQUARE_SIZE

            # Draw a semi-transparent rectangle over it to highlight
            highlight_surface = pygame.Surface(
                (SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA
            )
            highlight_surface.fill(
                (COLOR_HIGHLIGHT[0], COLOR_HIGHLIGHT[1], COLOR_HIGHLIGHT[2], 100)
            )
            self.screen.blit(highlight_surface, (rect_x, rect_y))
        logging.debug("Moves highlighted.")

    @eidosian()
    def get_square_under_mouse(self) -> Optional[int]:
        """
        Get the board square index (0..63 in python-chess coordinates) under the current mouse position.

        Returns:
            The square index under the mouse, or None if the mouse is outside the board.
        """
        mouse_x, mouse_y = pygame.mouse.get_pos()
        col = mouse_x // SQUARE_SIZE
        row = mouse_y // SQUARE_SIZE
        logging.debug(
            f"Mouse position: ({mouse_x}, {mouse_y}), Calculated col: {col}, row: {row}"
        )

        # Check if within board boundaries
        if col < 0 or col >= 8 or row < 0 or row >= 8:
            logging.debug("Mouse is outside the board.")
            return None

        # Convert row (from top) back to python-chess indexing
        # python-chess: 0..7 from top is rank 8 down to rank 1
        # row=0 is top, which is rank 8 => rank = 8 - row
        # square idx = (7-row)*8 + col (since 0..63 starts from top left is A8=0, B8=1,...)
        square_idx = (7 - row) * 8 + col
        logging.debug(f"Square index under mouse: {square_idx}")
        return square_idx

    @eidosian()
    def get_legal_moves_for_square(self, square: int) -> List[int]:
        """
        Return a list of target squares (integers) that the piece on 'square' can move to legally.

        Args:
            square: The index of the square to check for legal moves.

        Returns:
            A list of legal move target square indices.
        """
        moves = []
        logging.debug(f"Getting legal moves for square: {square}")
        # Check all legal moves from the board
        for move in self.board.legal_moves:
            # move.from_square is the starting square
            # move.to_square is the destination
            if move.from_square == square:
                moves.append(move.to_square)
        logging.debug(f"Legal moves for square {square}: {moves}")
        return moves

    @eidosian()
    def main_loop(self) -> None:
        """
        The main game loop that keeps the window open, handles events, draws the board, etc.
        """
        logging.info("Starting main game loop...")
        # We'll create a font for drawing piece symbols
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 48)  # 48px font
        logging.debug("Pygame font initialized.")

        clock = pygame.time.Clock()
        logging.debug("Pygame clock initialized.")

        # We'll store the last move for story generation
        last_move = None
        logging.debug("Last move initialized to None.")

        # Start the loop
        running = True
        story_generator = None

        while running:
            # Limit the frame rate to 30 FPS
            clock.tick(30)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logging.info("Quit event received. Exiting game loop.")
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # If the left mouse button is pressed, handle piece selection/move
                    if event.button == 1:
                        self.handle_click()
                    elif event.button == 4:  # Scroll up
                        self.text_processor.scroll_offset = max(
                            0, self.text_processor.scroll_offset - 20
                        )
                    elif event.button == 5:  # Scroll down
                        max_offset = len(self.rendered_lines) * 20 - PANEL_HEIGHT + 30
                        self.text_processor.scroll_offset = min(
                            max_offset, self.text_processor.scroll_offset + 20
                        )

            # Clear screen with background color
            self.screen.fill(COLOR_BACKGROUND)

            # Draw the board and pieces
            self.draw_board()

            # If we have a selected square, highlight its legal moves
            if self.selected_square is not None:
                highlight_squares = self.get_legal_moves_for_square(
                    self.selected_square
                )
                self.highlight_moves(highlight_squares)

            # Story generation handling
            if story_generator:
                try:
                    # Get complete story before proceeding
                    full_story = next(story_generator)
                    self.update_story(full_story)
                    pygame.display.update(self.text_panel)
                    story_generator = None
                except Exception as e:
                    logging.error(f"Story generation error: {e}")
                    story_generator = None

            # Update the display
            pygame.display.flip()

            # Check for game over
            if self.board.is_game_over():
                self.game_over = True
                result = self.board.result()  # e.g. '1-0', '0-1', or '1/2-1/2'
                logging.info(f"Game over with result: {result}")

                # Update player stats
                self.update_player_profile_stats(result)
                running = False

            # If it's the AI or story generation turn, you could handle that here
            # but for demonstration, it's purely user vs. user with story prompts.

            # (Optional) If you want to generate a story each turn (or on demand),
            # find which piece moved last and ask for a story.
            current_move_stack = self.board.move_stack
            if len(current_move_stack) > 0 and current_move_stack[-1] != last_move:
                # A new move occurred
                last_move = current_move_stack[-1]
                logging.debug(f"New move detected: {last_move}")
                # Identify which piece made that move
                moved_piece = self.board.piece_at(last_move.to_square)
                if moved_piece:
                    # Get the piece symbol (like 'P', 'q', etc.)
                    piece_symbol = moved_piece.symbol()
                    logging.debug(f"Moved piece: {piece_symbol}")
                    # Prepare possible moves for the next turn for that piece (in theory)
                    # In practice, the piece might not have the move again immediately,
                    # but let's demonstrate how you might do it:
                    story_moves = [str(m) for m in self.board.legal_moves]
                    # Generate a story with the local LLM or stub
                    story_generator = generate_llm_story(
                        self.board,
                        self.personalities,
                        last_move,
                        piece_symbol,
                        story_moves,
                    )
                    print("[Story Generated]", story_generator)
                    logging.info("Story generated and printed.")

        # When we exit the loop, we can do cleanup
        pygame.quit()
        logging.info("Pygame quit.")
        sys.exit()

    @eidosian()
    def handle_click(self) -> None:
        """
        Handle the user clicking on the board.
        """
        # Figure out which square (if any) is under the mouse
        clicked_square = self.get_square_under_mouse()
        if clicked_square is None:
            logging.debug("Click outside the board.")
            return

        logging.debug(f"Clicked square: {clicked_square}")
        # If we already have a selected square, we try to make a move
        if self.selected_square is not None:
            # Attempt to move from selected_square to clicked_square
            move = chess.Move(self.selected_square, clicked_square)
            logging.debug(f"Attempting move: {move}")
            if move in self.board.legal_moves:
                # Make the move on the board
                self.board.push(move)
                logging.debug(f"Move successful: {move}")
                # Reset selection
                self.selected_square = None
                logging.debug("Selected square reset to None.")
            else:
                # If it's not a legal move, just select the new square
                self.selected_square = clicked_square
                logging.debug(
                    f"Move illegal. Selected square updated to: {self.selected_square}"
                )
        else:
            # If no square was selected, just select the clicked square
            self.selected_square = clicked_square
            logging.debug(f"Selected square set to: {self.selected_square}")

    @eidosian()
    def update_player_profile_stats(self, result: str) -> None:
        """
        Update the current player's stats based on the game result.

        Args:
            result: The result of the game ('1-0', '0-1', or '1/2-1/2').
        """
        logging.info(f"Updating player profile stats with result: {result}")
        # Increment games played
        self.player_profiles[self.current_player_name]["games_played"] += 1
        logging.debug(f"Games played incremented for {self.current_player_name}")

        # If result is '1-0', white won
        # If result is '0-1', black won
        # If '1/2-1/2', it's a draw
        if result == "1-0":
            # If white is considered the current player (very naive assumption here)
            self.player_profiles[self.current_player_name]["wins"] += 1
            logging.debug(f"Win recorded for {self.current_player_name}")
        elif result == "0-1":
            # If black is considered the current player
            self.player_profiles[self.current_player_name]["losses"] += 1
            logging.debug(f"Loss recorded for {self.current_player_name}")
        else:
            # It's a draw or something else
            logging.debug(f"Draw recorded for {self.current_player_name}")
            pass

        # Save the updated profiles
        save_player_profiles(self.player_profiles)
        logging.info("Player profile stats updated and saved.")

    @eidosian()
    def wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text into multiple lines"""
        words = text.split()
        lines = []
        current_line: list[str] = []
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
        """Thread-safe story text update with processing"""
        if new_text != self.last_processed_text:
            self.last_processed_text = new_text
            threading.Thread(target=self._async_process_text, daemon=True).start()

    def _async_process_text(self):
        """Background text processing"""
        self.text_processor.process_text(self.story_text, self.personalities)
        self.rendered_lines = self._cache_text_surfaces()

    def _cache_text_surfaces(self) -> List[List[Dict]]:
        """Pre-render text segments with size caching"""
        rendered = []
        with self.text_processor.lock:
            for line in self.text_processor.current_lines:
                rendered_line = []
                x = 5
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
# MAIN ENTRY POINT
# -------------------------------------------------------------------------------------


@eidosian()
def main() -> None:
    """
    The main entry point that initializes Pygame, creates the PersonalityChessGame, and starts it.
    """
    logging.info("Starting main function...")
    # Initialize Pygame
    pygame.init()
    logging.debug("Pygame initialized.")

    # Get display info and set fullscreen
    info = pygame.display.Info()
    global SQUARE_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT
    SQUARE_SIZE = min(info.current_w // 10, info.current_h // 10)
    SCREEN_WIDTH = info.current_w
    SCREEN_HEIGHT = info.current_h

    # Create fullscreen window
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Fun Chess Game Girls - Personality Chess Prototype")
    logging.debug("Pygame screen and caption set.")

    warmup_llm_connection()  # Add this after pygame.init()
    game = PersonalityChessGame(screen, DEFAULT_PERSONALITIES)
    logging.debug("PersonalityChessGame object created.")

    # Run the main loop
    game.main_loop()
    logging.info("Main function finished.")


# Run main if this file is executed directly
if __name__ == "__main__":
    main()
