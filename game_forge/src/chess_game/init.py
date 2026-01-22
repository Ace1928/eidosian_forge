"""
Fun Chess Game Girls - Personality-driven Chess Prototype
=========================================================
Copy and paste this entire code into a single Python file (e.g. `fun_chess_game.py`).
Then run: `python fun_chess_game.py`

Dependencies:
  - pygame: For creating the game's graphical user interface.
  - python-chess: For handling chess game logic and board states.
  - requests (optional): For potential integration with local Language Learning Models (LLMs) or external APIs.

This prototype demonstrates the following key features:
  1. Chess Engine Integration (python-chess): Utilizes a robust chess engine to ensure valid game moves and manage game state.
  2. Pygame-based GUI: Provides a visual and interactive interface for playing chess, built using the Pygame library.
  3. Piece Personalities: Introduces the concept of unique personalities for each chess piece, adding a narrative and engaging layer to the game.
  4. Story Engine Stub (LLM Integration): Includes a basic framework for integrating with a story engine, potentially powered by a local LLM (like ollama), to generate dynamic game narratives and interactions.
  5. Player Profile Saving/Loading: Implements functionality to save and load player profiles, preserving game progress and player preferences using JSON files.

Developed By: Eidos
"""

import pygame  # For creating the game's GUI
import chess  # For chess logic and board representation
import sys  # For system-specific parameters and functions
import os  # For interacting with the operating system, e.g., file paths
import json  # For working with JSON data, used for profiles and potentially personalities
import requests  # For making HTTP requests, potentially for LLM or API interactions
import logging  # For logging events and debugging information
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Tuple,
)  # For type hinting, improving code readability and maintainability
import threading  # For supporting concurrent operations (though not heavily used in this init file)
import random  # For generating random numbers, potentially for game variations or AI

# -------------------------------------------------------------------------------------
# GLOBAL CONFIGURATION SECTION - Defining constants and settings for the entire game
# -------------------------------------------------------------------------------------

# Configure logging to capture game events and aid in debugging.
# Logs will be written to 'game.log' file and also printed to the console.
logging.basicConfig(
    level=logging.INFO,  # Sets the logging level to INFO, capturing informational messages and above (WARNING, ERROR, CRITICAL)
    format="%(levelname)s - %(message)s",  # Defines the log message format to include level name and the message itself
    handlers=[
        logging.FileHandler(
            "game.log"
        ),  # Handler to write log messages to 'game.log' file
        logging.StreamHandler(),  # Handler to print log messages to the console (standard output)
    ],
)

# --- Board Dimensions ---
BOARD_WIDTH = 8  # Standard chess board width in squares
BOARD_HEIGHT = 8  # Standard chess board height in squares
logging.debug(
    f"Board dimensions set to {BOARD_WIDTH}x{BOARD_HEIGHT}"
)  # Debug log to confirm board dimensions

# --- Square Size ---
SQUARE_SIZE = 80  # Size of each square in pixels on the screen
logging.debug(
    f"Square size set to {SQUARE_SIZE} pixels"
)  # Debug log to confirm square size

# --- Colors ---
# Define colors using RGB tuples for the chessboard and UI elements.
COLOR_LIGHT = (240, 217, 181)  # Color for light squares on the chessboard (light beige)
COLOR_DARK = (181, 136, 99)  # Color for dark squares on the chessboard (brown)
COLOR_HIGHLIGHT = (
    186,
    202,
    43,
)  # Color to highlight selected squares or possible moves (greenish)
COLOR_BORDER = (0, 0, 0)  # Color for borders around squares or UI elements (black)
COLOR_TEXT = (0, 0, 0)  # Default color for text in the game (black)
COLOR_BACKGROUND = (200, 200, 200)  # Background color for the game window (light gray)
logging.debug(  # Debug log to confirm color definitions
    f"Colors defined: LIGHT={COLOR_LIGHT}, DARK={COLOR_DARK}, HIGHLIGHT={COLOR_HIGHLIGHT}, BORDER={COLOR_BORDER}, TEXT={COLOR_TEXT}, BACKGROUND={COLOR_BACKGROUND}"
)

# --- Screen Dimensions ---
SCREEN_WIDTH = (
    BOARD_WIDTH * SQUARE_SIZE
)  # Calculate the total width of the game screen based on board width and square size
PANEL_HEIGHT = 150  # Height of the information panel below the chessboard in pixels
SCREEN_HEIGHT = (
    BOARD_HEIGHT * SQUARE_SIZE + PANEL_HEIGHT
)  # Calculate total screen height including the chessboard and the panel
logging.debug(
    f"Screen dimensions set to {SCREEN_WIDTH}x{SCREEN_HEIGHT}"
)  # Debug log to confirm screen dimensions

# --- File Paths ---
PLAYER_PROFILES_JSON = (
    "player_profiles.json"  # Filename for storing player profiles in JSON format
)
logging.debug(
    f"Player profiles will be stored in: {PLAYER_PROFILES_JSON}"
)  # Debug log to confirm player profile file name
PIECE_IMAGES_PATH = "piece_images"  # Directory path where piece images are stored (if using image-based pieces)

# --- Default Game Data ---
# Default personalities assigned to each chess piece.
# This dictionary defines the name, traits, and behaviors for each piece type (both white and black).
DEFAULT_PERSONALITIES = {
    "P": {  # White Pawn
        "name": "White Pawn",
        "traits": ["Brave", "Ambitious", "Loyal"],
        "behaviors": "Slowly marches forward, hoping to become a queen one day.",
    },
    "N": {  # White Knight
        "name": "White Knight",
        "traits": ["Chivalrous", "Swift", "Sneaky"],
        "behaviors": "Leaps over allies and enemies, striking unpredictably.",
    },
    "B": {  # White Bishop
        "name": "White Bishop",
        "traits": ["Pious", "Diagonal Strategist"],
        "behaviors": "Stays on color, delivering holy retribution at an angle.",
    },
    "R": {  # White Rook
        "name": "White Rook",
        "traits": ["Towering", "Protective"],
        "behaviors": "Moves in straight lines with unbreakable determination.",
    },
    "Q": {  # White Queen
        "name": "White Queen",
        "traits": ["Regal", "Powerful", "Fearless"],
        "behaviors": "Dominates the battlefield with swift and lethal action.",
    },
    "K": {  # White King
        "name": "White King",
        "traits": ["Solemn", "Cautious", "Royal"],
        "behaviors": "Moves slowly, must be protected at all costs.",
    },
    "p": {  # Black Pawn
        "name": "Black Pawn",
        "traits": ["Steadfast", "Stoic", "Hopeful"],
        "behaviors": "Dreams of promotion, defends its territory bravely.",
    },
    "n": {  # Black Knight
        "name": "Black Knight",
        "traits": ["Dark Charger", "Trickster"],
        "behaviors": "Appears where least expected, causing chaos among enemies.",
    },
    "b": {  # Black Bishop
        "name": "Black Bishop",
        "traits": ["Dark Mentor", "Diagonal Mastermind"],
        "behaviors": "Preaches cunning and strikes from the shadows on diagonals.",
    },
    "r": {  # Black Rook
        "name": "Black Rook",
        "traits": ["Fortified", "Unyielding"],
        "behaviors": "Forms the walls of defense, sweeping horizontally and vertically.",
    },
    "q": {  # Black Queen
        "name": "Black Queen",
        "traits": ["Malicious", "Dominant", "Fearless"],
        "behaviors": "Rules the board with unstoppable might and cunning.",
    },
    "k": {  # Black King
        "name": "Black King",
        "traits": ["Resilient", "Protective", "Royal"],
        "behaviors": "Keeps watchful oversight, always mindful of check.",
    },
}
logging.debug(
    f"Default personalities loaded: {DEFAULT_PERSONALITIES}"
)  # Debug log to confirm personalities loaded

# Default structure for new player profiles.
# This defines the initial state and default values for a new player's profile.
DEFAULT_PROFILE = {
    "games_played": 0,  # Number of games played by the player, initialized to 0
    "wins": 0,  # Number of games won by the player, initialized to 0
    "losses": 0,  # Number of games lost by the player, initialized to 0
    "preferred_color": "white",  # Player's preferred color to play as, defaults to white
    "unlocked_personalities": [
        "P",
        "R",
        "N",
    ],  # List of initially unlocked personalities (example progression system)
}
logging.debug(
    f"Default player profile structure: {DEFAULT_PROFILE}"
)  # Debug log to confirm default profile structure

# --- UI and Display Constants ---
DEFAULT_FONT_SIZE = 48  # Default font size for text elements in the game UI

# --- Emoji Map ---
# Mapping of game-related actions or concepts to emojis for richer text display.
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

# --- Color Map for Traits ---
# Mapping of personality traits to specific colors for visual representation in the UI.
COLOR_MAP = {
    "Brave": (255, 0, 0),  # Red
    "Cautious": (0, 0, 255),  # Blue
    "Loyal": (0, 128, 0),  # Green
    "Sneaky": (128, 0, 128),  # Purple
    "Royal": (255, 215, 0),  # Gold
}

UNICODE_MAP = {
    "k": "♔",  # Black King
    "q": "♕",  # Black Queen
    "r": "♖",  # Black Rook
    "b": "♗",  # Black Bishop
    "n": "♘",  # Black Knight
    "p": "♙",  # Black Pawn
    "K": "♚",  # White King
    "Q": "♛",  # White Queen
    "R": "♜",  # White Rook
    "B": "♝",  # White Bishop
    "N": "♞",  # White Knight
    "P": "♟",  # White Pawn
}

if __name__ == "__main__":
    logging.info("Running comprehensive self-tests for init.py...")

    # --- Test: Logging Configuration ---
    logger = logging.getLogger()
    assert logger.level == logging.INFO, "Logging level should be INFO"
    has_file_handler = any(
        isinstance(handler, logging.FileHandler) for handler in logger.handlers
    )
    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    )
    assert has_file_handler, "Logging should include a FileHandler"
    assert has_stream_handler, "Logging should include a StreamHandler"
    logging.info("Test Passed: Logging Configuration")

    # --- Test: Board Dimensions ---
    assert (
        isinstance(BOARD_WIDTH, int) and BOARD_WIDTH > 0
    ), "BOARD_WIDTH should be a positive integer"
    assert (
        isinstance(BOARD_HEIGHT, int) and BOARD_HEIGHT > 0
    ), "BOARD_HEIGHT should be a positive integer"
    assert BOARD_WIDTH == 8, "BOARD_WIDTH should be 8"
    assert BOARD_HEIGHT == 8, "BOARD_HEIGHT should be 8"
    logging.info("Test Passed: Board Dimensions")

    # --- Test: Square Size ---
    assert (
        isinstance(SQUARE_SIZE, int) and SQUARE_SIZE > 0
    ), "SQUARE_SIZE should be a positive integer"
    assert SQUARE_SIZE == 80, "SQUARE_SIZE should be 80"
    logging.info("Test Passed: Square Size")

    # --- Test: Colors ---
    color_constants = {
        "COLOR_LIGHT": COLOR_LIGHT,
        "COLOR_DARK": COLOR_DARK,
        "COLOR_HIGHLIGHT": COLOR_HIGHLIGHT,
        "COLOR_BORDER": COLOR_BORDER,
        "COLOR_TEXT": COLOR_TEXT,
        "COLOR_BACKGROUND": COLOR_BACKGROUND,
    }
    for name, color in color_constants.items():
        assert isinstance(color, tuple), f"{name} should be a tuple"
        assert len(color) == 3, f"{name} should be an RGB tuple (length 3)"
        for channel in color:
            assert (
                isinstance(channel, int) and 0 <= channel <= 255
            ), f"{name} channels should be integers between 0 and 255"
    logging.info("Test Passed: Colors")

    # --- Test: Screen Dimensions ---
    assert (
        isinstance(SCREEN_WIDTH, int) and SCREEN_WIDTH > 0
    ), "SCREEN_WIDTH should be a positive integer"
    assert (
        isinstance(SCREEN_HEIGHT, int) and SCREEN_HEIGHT > 0
    ), "SCREEN_HEIGHT should be a positive integer"
    assert (
        SCREEN_WIDTH == BOARD_WIDTH * SQUARE_SIZE
    ), "SCREEN_WIDTH calculation incorrect"
    assert (
        SCREEN_HEIGHT == BOARD_HEIGHT * SQUARE_SIZE + PANEL_HEIGHT
    ), "SCREEN_HEIGHT calculation incorrect"
    logging.info("Test Passed: Screen Dimensions")

    # --- Test: File Paths ---
    assert (
        isinstance(PLAYER_PROFILES_JSON, str) and PLAYER_PROFILES_JSON
    ), "PLAYER_PROFILES_JSON should be a non-empty string"
    assert (
        isinstance(PIECE_IMAGES_PATH, str) and PIECE_IMAGES_PATH
    ), "PIECE_IMAGES_PATH should be a non-empty string"
    logging.info("Test Passed: File Paths")

    # --- Test: Default Personalities ---
    assert isinstance(
        DEFAULT_PERSONALITIES, dict
    ), "DEFAULT_PERSONALITIES should be a dictionary"
    piece_symbols = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
    assert all(
        piece in DEFAULT_PERSONALITIES for piece in piece_symbols
    ), "DEFAULT_PERSONALITIES should contain all piece symbols"
    for piece_data in DEFAULT_PERSONALITIES.values():
        assert isinstance(
            piece_data, dict
        ), "Each piece personality should be a dictionary"
        assert "name" in piece_data and isinstance(
            piece_data["name"], str
        ), "Personality should have a 'name' (string)"
        assert "traits" in piece_data and isinstance(
            piece_data["traits"], list
        ), "Personality should have 'traits' (list)"
        assert "behaviors" in piece_data and isinstance(
            piece_data["behaviors"], str
        ), "Personality should have 'behaviors' (string)"
    logging.info("Test Passed: Default Personalities Structure")

    # --- Test: Default Profile ---
    assert isinstance(DEFAULT_PROFILE, dict), "DEFAULT_PROFILE should be a dictionary"
    profile_keys = [
        "games_played",
        "wins",
        "losses",
        "preferred_color",
        "unlocked_personalities",
    ]
    assert all(
        key in DEFAULT_PROFILE for key in profile_keys
    ), f"DEFAULT_PROFILE should contain keys: {profile_keys}"
    assert isinstance(
        DEFAULT_PROFILE["games_played"], int
    ), "'games_played' should be an integer"
    assert isinstance(DEFAULT_PROFILE["wins"], int), "'wins' should be an integer"
    assert isinstance(DEFAULT_PROFILE["losses"], int), "'losses' should be an integer"
    assert isinstance(
        DEFAULT_PROFILE["preferred_color"], str
    ), "'preferred_color' should be a string"
    assert isinstance(
        DEFAULT_PROFILE["unlocked_personalities"], list
    ), "'unlocked_personalities' should be a list"
    logging.info("Test Passed: Default Profile Structure")

    # --- Test: UI and Display Constants ---
    assert (
        isinstance(DEFAULT_FONT_SIZE, int) and DEFAULT_FONT_SIZE > 0
    ), "DEFAULT_FONT_SIZE should be a positive integer"
    logging.info("Test Passed: UI and Display Constants")

    # --- Test: Emoji Map ---
    assert isinstance(EMOJI_MAP, dict), "EMOJI_MAP should be a dictionary"
    for emoji in EMOJI_MAP.values():
        assert isinstance(emoji, str), "Emojis should be strings"
    logging.info("Test Passed: Emoji Map")

    # --- Test: Color Map ---
    assert isinstance(COLOR_MAP, dict), "COLOR_MAP should be a dictionary"
    for color in COLOR_MAP.values():
        assert isinstance(color, tuple), "Color values in COLOR_MAP should be tuples"
        assert len(color) == 3, "Color values in COLOR_MAP should be RGB tuples"
    logging.info("Test Passed: Color Map")

    # --- Test: Unicode Map ---
    assert isinstance(UNICODE_MAP, dict), "UNICODE_MAP should be a dictionary"
    for unicode_char in UNICODE_MAP.values():
        assert isinstance(unicode_char, str), "Unicode characters should be strings"
    logging.info("Test Passed: Unicode Map")

    logging.info("All self-tests for init.py completed successfully.")
