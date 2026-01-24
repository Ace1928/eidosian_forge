import pygame
import threading
import re
import logging
from typing import Dict, List, Any, Tuple, Optional

from init import EMOJI_MAP, COLOR_MAP, PANEL_HEIGHT
from eidosian_core import eidosian

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TextSegment:
    def __init__(
        self, text: str, color: Tuple[int, int, int], space_after: bool = True
    ):
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, not {type(text)}")
        if not (
            isinstance(color, tuple)
            and len(color) == 3
            and all(isinstance(c, int) for c in color)
        ):
            raise TypeError(f"color must be a tuple of 3 ints, not {color}")
        if not isinstance(space_after, bool):
            raise TypeError(f"space_after must be a bool, not {type(space_after)}")
        self.text = text
        self.color = color
        self.space_after = space_after

    def __repr__(self) -> str:
        return f"TextSegment(text='{self.text}', color={self.color}, space_after={self.space_after})"

    def __eq__(self, other):
        return (
            isinstance(other, TextSegment)
            and self.text == other.text
            and self.color == other.color
            and self.space_after == other.space_after
        )


class TextLine:
    def __init__(self, segments: Optional[List[TextSegment]] = None):
        if segments is not None and not isinstance(segments, list):
            raise TypeError(f"segments must be a list or None, not {type(segments)}")
        if segments and not all(isinstance(seg, TextSegment) for seg in segments):
            raise TypeError("segments must contain TextSegment")
        self.segments: List[TextSegment] = segments if segments else []

    @eidosian()
    def add_segment(self, segment: TextSegment) -> None:
        if not isinstance(segment, TextSegment):
            raise TypeError(f"segment must be a TextSegment, not {type(segment)}")
        self.segments.append(segment)

    def __repr__(self) -> str:
        return f"TextLine(segments={self.segments})"

    def __eq__(self, other):
        return isinstance(other, TextLine) and self.segments == other.segments


class TextProcessor:
    def __init__(self, max_width: int, font: pygame.font.Font) -> None:
        if not isinstance(max_width, int) or max_width <= 0:
            logging.error(f"Invalid max_width: {max_width}. Must be positive integer.")
            raise ValueError("max_width must be a positive integer.")
        if not isinstance(font, pygame.font.Font):
            logging.error(f"Invalid font: {font}. Must be a pygame.font.Font object.")
            raise ValueError("font must be a pygame.font.Font object.")

        self.max_width: int = max_width
        self.font: pygame.font.Font = font
        self.current_lines: List[TextLine] = []
        self.scroll_offset: int = 0
        self.lock: threading.Lock = threading.Lock()
        self.cache: Dict[
            Tuple[str, Tuple[int, int, int], pygame.font.Font], pygame.Surface
        ] = {}
        self.emoji_pattern: re.Pattern = self._build_emoji_regex()
        self.line_height: int = self.font.get_height() + 2
        self.space_width: int = self.font.size(" ")[0]
        logging.info("TextProcessor initialized.")

    @eidosian()
    def process_text(
        self, raw_text: str, personalities: Optional[Dict[str, Any]] = None
    ) -> None:
        if not isinstance(raw_text, str):
            logging.error(f"Invalid input text type: {type(raw_text)}. Expected str.")
            raise TypeError("raw_text must be a string.")
        if personalities is not None and not isinstance(personalities, dict):
            logging.error(
                f"Invalid personalities type: {type(personalities)}. Expected Dict or None."
            )
            raise TypeError("personalities must be a dictionary or None.")

        logging.debug("Starting text processing.")
        with self.lock:
            try:
                processed_text = self._add_emojis(raw_text)
                segments = (
                    self._apply_semantic_coloring(processed_text, personalities)
                    if personalities
                    else self._segment_text(processed_text)
                )
                self.current_lines = self._context_aware_wrap(segments)
                self._auto_scroll()
                logging.debug("Text processing completed successfully.")
            except Exception as e:
                logging.error(f"Error during text processing: {e}", exc_info=True)
                raise

    def _build_emoji_regex(self) -> re.Pattern:
        if not EMOJI_MAP:
            logging.warning("EMOJI_MAP is empty. Emoji processing will be skipped.")
            return re.compile("")
        escaped_keys = [re.escape(key) for key in EMOJI_MAP.keys()]
        emoji_regex_pattern = r"\b(" + "|".join(escaped_keys) + r")\b"
        logging.debug(f"Emoji regex pattern built: {emoji_regex_pattern}")
        return re.compile(emoji_regex_pattern, re.IGNORECASE)

    def _add_emojis(self, text: str) -> str:
        if not self.emoji_pattern:
            return text

        @eidosian()
        def replace_emoji(match: re.Match) -> str:
            keyword = match.group(0).lower()
            if emoji := EMOJI_MAP.get(keyword):
                return f"{match.group(0)} {emoji}".strip()
            logging.warning(f"Emoji not found for keyword: '{keyword}' in EMOJI_MAP.")
            return match.group(0)

        processed_text = self.emoji_pattern.sub(replace_emoji, text)
        logging.debug("Emojis added to text.")
        return processed_text

    def _apply_semantic_coloring(
        self, text: str, personalities: Optional[Dict[str, Any]]
    ) -> List[TextSegment]:
        if not personalities or not personalities.get("traits"):
            logging.info(
                "No traits provided or traits list is empty. Applying default segmentation."
            )
            return self._segment_text(text)

        traits = personalities.get("traits", [])
        trait_priority = {trait: index for index, trait in enumerate(traits)}
        tokens = self._tokenize_text(text)

        @eidosian()
        def get_segment_color(token: str) -> Tuple[int, int, int]:
            best_match_trait = None
            highest_priority = -1
            for trait in traits:
                if re.search(r"\b" + re.escape(trait) + r"\b", token, re.IGNORECASE):
                    priority = trait_priority.get(trait, -1)
                    if priority > highest_priority:
                        highest_priority = priority
                        best_match_trait = trait
            if best_match_trait:
                color = COLOR_MAP.get(best_match_trait, (0, 0, 0))
                if color == (0, 0, 0):
                    logging.warning(
                        f"Color not found for trait: '{best_match_trait}' in COLOR_MAP. Using default color."
                    )
                return color
            return (0, 0, 0)

        colored_segments = [
            TextSegment(token, get_segment_color(token), not token.isspace())
            for token in tokens
        ]

        for i in range(len(colored_segments) - 1):
            if colored_segments[i].text.isspace():
                colored_segments[i].space_after = False
                if i > 0:
                    colored_segments[i - 1].space_after = True
            else:
                colored_segments[i].space_after = True

        if colored_segments and not colored_segments[-1].text.isspace():
            colored_segments[-1].space_after = False

        logging.debug("Semantic coloring applied to text.")
        return colored_segments

    def _segment_text(self, text: str) -> List[TextSegment]:
        tokens = self._tokenize_text(text)
        segments = [
            TextSegment(token, (0, 0, 0), not token.isspace()) for token in tokens
        ]

        for i in range(len(segments) - 1):
            if segments[i].text.isspace():
                segments[i].space_after = False
                if i > 0:
                    segments[i - 1].space_after = True
            else:
                segments[i].space_after = True

        if segments and not segments[-1].text.isspace():
            segments[-1].space_after = False
        return segments

    def _tokenize_text(self, text: str) -> List[str]:
        tokens = []
        current_token = ""
        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                tokens.append(char)
                current_token = ""
            elif re.match(r"[\w'’\-]+", char):
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                tokens.append(char)
                current_token = ""
        if current_token:
            tokens.append(current_token)
        return tokens

    def _context_aware_wrap(self, segments: List[TextSegment]) -> List[TextLine]:
        lines = []
        current_line = TextLine()
        current_line_width = 0

        for segment in segments:
            text = segment.text
            color = segment.color
            space_width = self.space_width if segment.space_after else 0
            segment_width = self.font.size(text)[0]
            required_width = segment_width + space_width

            if (
                current_line.segments
                and current_line_width + required_width > self.max_width
            ):
                lines.append(current_line)
                current_line = TextLine()
                current_line_width = 0

            if segment_width > self.max_width and len(text) > 1:
                hyphenated_segments = self._hyphenate_word(text, color)
                for sub_segment in hyphenated_segments:
                    sub_segment_width = self.font.size(sub_segment.text)[0]
                    if (
                        current_line.segments
                        and current_line_width + sub_segment_width > self.max_width
                    ):
                        lines.append(current_line)
                        current_line = TextLine()
                        current_line_width = 0
                    current_line.add_segment(sub_segment)
                    current_line_width += sub_segment_width
            else:
                current_line.add_segment(segment)
                current_line_width += segment_width

            if segment.space_after:
                current_line_width += space_width

        if current_line.segments:
            lines.append(current_line)
        logging.debug("Text wrapping completed.")
        return lines

    def _hyphenate_word(
        self, word: str, color: Tuple[int, int, int]
    ) -> List[TextSegment]:
        segments = []
        current_part = ""
        current_width = 0
        max_part_width = self.max_width - self.space_width

        for char in word:
            char_width = self.font.size(char)[0]
            if current_width + char_width > max_part_width:
                segments.append(TextSegment(current_part + "-", color, False))
                current_part = char
                current_width = char_width
            else:
                current_part += char
                current_width += char_width
        segments.append(TextSegment(current_part, color, False))
        return segments

    def _auto_scroll(self) -> None:
        text_height = len(self.current_lines) * self.line_height
        self.scroll_offset = max(0, text_height - PANEL_HEIGHT)
        logging.debug(f"Auto scroll adjusted. Scroll offset: {self.scroll_offset}")

    @eidosian()
    def render_lines(self, surface: pygame.Surface, start_y: int) -> None:
        y_offset = start_y - self.scroll_offset
        try:
            for line in self.current_lines:
                x_offset = 0
                for segment in line.segments:
                    rendered_text = self._get_rendered_text(segment.text, segment.color)
                    surface.blit(rendered_text, (x_offset, y_offset))
                    x_offset += rendered_text.get_width()
                y_offset += self.line_height
        except Exception as e:
            logging.error(f"Error during text rendering: {e}", exc_info=True)
            raise

    def _get_rendered_text(
        self, text: str, color: Tuple[int, int, int]
    ) -> pygame.Surface:
        cache_key = (text, color, self.font)
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            surface = self.font.render(text, True, color)
            self.cache[cache_key] = surface
            return surface
        except Exception as e:
            logging.error(f"Error rendering text: '{text}'. {e}", exc_info=True)
            return pygame.Surface((0, 0))


if __name__ == "__main__":
    pygame.init()
    font = pygame.font.Font(None, 24)
    max_width = 400
    screen_width = 600
    screen_height = 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    # Example color map with more nuanced shades
    COLOR_MAP = {
        "happy": (0, 200, 0),
        "sad": (0, 0, 200),
        "angry": (200, 0, 0),
        "default": (50, 50, 50),
        "emphasis": (200, 200, 0),
        "link": (0, 200, 200),
        "happy_shade": (100, 255, 100),
        "sad_shade": (100, 100, 255),
        "angry_shade": (255, 100, 100),
    }

    # Example emoji map
    EMOJI_MAP = {
        ":smile:": "😊",
        ":heart:": "❤️",
        ":thumbsup:": "👍",
        ":exclamation:": "❗",
        ":thinking:": "🤔",
        ":fire:": "🔥",
        ":star:": "⭐",
        ":check:": "✅",
        ":cross:": "❌",
    }

    text_processor = TextProcessor(max_width, font)

    @eidosian()
    def apply_nuanced_coloring(text, personalities):
        """Applies nuanced coloring based on personalities and context."""
        if not personalities:
            return [(text, COLOR_MAP["default"])]

        colored_segments = []
        words = text.split()
        for word in words:
            color = COLOR_MAP["default"]
            if "traits" in personalities:
                if word in personalities["traits"]:
                    if word == "happy":
                        color = COLOR_MAP["happy"]
                    elif word == "sad":
                        color = COLOR_MAP["sad"]
                    elif word == "angry":
                        color = COLOR_MAP["angry"]
                elif word.endswith("ly"):
                    color = COLOR_MAP["emphasis"]
                elif word.startswith("un"):
                    color = COLOR_MAP["sad_shade"]
                elif word.isupper():
                    color = COLOR_MAP["angry_shade"]
                elif word.islower():
                    color = COLOR_MAP["happy_shade"]
            if "tags" in personalities:
                for tag, tag_name in personalities["tags"].items():
                    if f"<{tag}>" in word and f"</{tag}>" in word:
                        word = word.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
                        color = COLOR_MAP[tag_name]
            colored_segments.append((word, color))
        return colored_segments

    @eidosian()
    def run_test_case(test_name, raw_text, personalities=None, expected_output=None):
        screen.fill((255, 255, 255))
        print(f"\n--- {test_name} ---")
        print(f"Input: {raw_text}, Personalities: {personalities}")
        text_processor.process_text(raw_text, personalities)
        print("Processed Text:")
        for line in text_processor.current_lines:
            print(line)

        if expected_output:
            print("\nExpected Output:")
            for expected_line in expected_output:
                print(expected_line)

            actual_lines = [str(line) for line in text_processor.current_lines]
            expected_lines = [str(line) for line in expected_output]
            if actual_lines == expected_lines:
                print("\nTest Passed: Actual output matches expected output.")
            else:
                print("\nTest Failed: Actual output does not match expected output.")
                print("Actual Output:")
                for line in text_processor.current_lines:
                    print(line)
        else:
            print("\nNo expected output provided for comparison.")
        text_processor.render_lines(screen, 20)
        pygame.display.flip()
        pygame.time.delay(10000)  # Display for 10 seconds

    # Test cases
    test_cases = [
        (
            "Test Case 1 - Basic Text",
            "This is a test message. It should wrap correctly.",
            None,
            [
                TextLine(
                    [
                        TextSegment(text="This", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="is", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="a", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="test", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="message", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=".", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="It", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="should", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="wrap", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                    ]
                ),
                TextLine(
                    [
                        TextSegment(
                            text="correctly", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=".", color=(50, 50, 50), space_after=False),
                    ]
                ),
            ],
        ),
        (
            "Test Case 2 - Text with Emojis",
            "Hello :smile: and :heart:! Also :thumbsup: and :exclamation: :thinking: :fire: :star: :check: :cross:",
            None,
            [
                TextLine(
                    [
                        TextSegment(text="Hello", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="smile", color=(50, 50, 50), space_after=True),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="and", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="heart", color=(50, 50, 50), space_after=True),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="!", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="Also", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(
                            text="thumbsup", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                    ]
                ),
                TextLine(
                    [
                        TextSegment(text="and", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(
                            text="exclamation", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(
                            text="thinking", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="fire", color=(50, 50, 50), space_after=True),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="star", color=(50, 50, 50), space_after=True),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                    ]
                ),
                TextLine(
                    [
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="check", color=(50, 50, 50), space_after=True),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="cross", color=(50, 50, 50), space_after=True),
                        TextSegment(text=":", color=(50, 50, 50), space_after=False),
                    ]
                ),
            ],
        ),
        (
            "Test Case 3 - Text with Semantic Coloring",
            "I am feeling happy and a bit sad, but not angry.",
            {"traits": ["happy", "sad", "angry"]},
            [
                TextLine(
                    [
                        TextSegment(text="I", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="am", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="feeling", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="happy", color=(0, 200, 0), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="and", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="a", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="bit", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="sad", color=(0, 0, 200), space_after=True),
                        TextSegment(text=",", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="but", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="not", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                    ]
                ),
                TextLine(
                    [
                        TextSegment(text="angry", color=(200, 0, 0), space_after=True),
                        TextSegment(text=".", color=(50, 50, 50), space_after=False),
                    ]
                ),
            ],
        ),
        (
            "Test Case 4 - Long Word Hyphenation",
            "This is a verylongwordthatneedshyphenation.",
            None,
            [
                TextLine(
                    [
                        TextSegment(text="This", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="is", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="a", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="verylongwordthatneedshyphenation",
                            color=(50, 50, 50),
                            space_after=True,
                        ),
                        TextSegment(text=".", color=(50, 50, 50), space_after=False),
                    ]
                )
            ],
        ),
        (
            "Test Case 5 - Leading and Trailing Spaces",
            "   Leading and trailing spaces   ",
            None,
            [
                TextLine(
                    [
                        TextSegment(text=" ", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="Leading", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="and", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="trailing", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="spaces", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                    ]
                )
            ],
        ),
        ("Test Case 6 - Empty Text", "", None, []),
        (
            "Test Case 7 - Special Characters",
            r"Special characters: !@#$%^&*()_+=-`~[]\{}|;':\",./<>?",
            None,
            [
                TextLine(
                    [
                        TextSegment(
                            text="Special", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="characters", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="!", color=(50, 50, 50), space_after=True),
                        TextSegment(text="@", color=(50, 50, 50), space_after=True),
                        TextSegment(text="#", color=(50, 50, 50), space_after=True),
                        TextSegment(text="$", color=(50, 50, 50), space_after=True),
                        TextSegment(text="%", color=(50, 50, 50), space_after=True),
                        TextSegment(text="^", color=(50, 50, 50), space_after=True),
                        TextSegment(text="&", color=(50, 50, 50), space_after=True),
                        TextSegment(text="*", color=(50, 50, 50), space_after=True),
                        TextSegment(text="(", color=(50, 50, 50), space_after=True),
                        TextSegment(text=")", color=(50, 50, 50), space_after=True),
                        TextSegment(text="_", color=(50, 50, 50), space_after=True),
                        TextSegment(text="+", color=(50, 50, 50), space_after=True),
                        TextSegment(text="=", color=(50, 50, 50), space_after=True),
                        TextSegment(text="-", color=(50, 50, 50), space_after=True),
                        TextSegment(text="`", color=(50, 50, 50), space_after=True),
                        TextSegment(text="~", color=(50, 50, 50), space_after=True),
                        TextSegment(text="[", color=(50, 50, 50), space_after=True),
                        TextSegment(text="]", color=(50, 50, 50), space_after=True),
                    ]
                ),
                TextLine(
                    [
                        TextSegment(text="\\", color=(50, 50, 50), space_after=True),
                        TextSegment(text="{", color=(50, 50, 50), space_after=True),
                        TextSegment(text="}", color=(50, 50, 50), space_after=True),
                        TextSegment(text="|", color=(50, 50, 50), space_after=True),
                        TextSegment(text=";", color=(50, 50, 50), space_after=True),
                        TextSegment(text="'", color=(50, 50, 50), space_after=True),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="\\", color=(50, 50, 50), space_after=True),
                        TextSegment(text='"', color=(50, 50, 50), space_after=True),
                        TextSegment(text=",", color=(50, 50, 50), space_after=True),
                        TextSegment(text=".", color=(50, 50, 50), space_after=True),
                        TextSegment(text="/", color=(50, 50, 50), space_after=True),
                        TextSegment(text="<", color=(50, 50, 50), space_after=True),
                        TextSegment(text=">", color=(50, 50, 50), space_after=True),
                        TextSegment(text="?", color=(50, 50, 50), space_after=False),
                    ]
                ),
            ],
        ),
        (
            "Test Case 8 - Multiple Spaces",
            "Multiple   spaces   here.",
            None,
            [
                TextLine(
                    [
                        TextSegment(
                            text="Multiple", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="spaces", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="here", color=(50, 50, 50), space_after=True),
                        TextSegment(text=".", color=(50, 50, 50), space_after=False),
                    ]
                )
            ],
        ),
        (
            "Test Case 9 - Mixed Cases",
            "MiXeD CaSeS TeSt.",
            None,
            [
                TextLine(
                    [
                        TextSegment(
                            text="MiXeD", color=(255, 100, 100), space_after=True
                        ),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="CaSeS", color=(255, 100, 100), space_after=True
                        ),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="TeSt", color=(255, 100, 100), space_after=True
                        ),
                        TextSegment(text=".", color=(50, 50, 50), space_after=False),
                    ]
                )
            ],
        ),
        (
            "Test Case 10 - Numbers",
            "Numbers: 1234567890.",
            None,
            [
                TextLine(
                    [
                        TextSegment(
                            text="Numbers", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="1234567890", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=".", color=(50, 50, 50), space_after=False),
                    ]
                )
            ],
        ),
        (
            "Test Case 11 - Mixed Test",
            "This is a :smile: mixed test with some happy and sad feelings, and a verylongwordthatneedshyphenation. 12345 and :heart:! :fire: :star:",
            {"traits": ["happy", "sad"]},
            [
                TextLine(
                    [
                        TextSegment(text="This", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="is", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="a", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="smile", color=(50, 50, 50), space_after=True),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="mixed", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="test", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="with", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="some", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="happy", color=(0, 200, 0), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                    ]
                ),
                TextLine(
                    [
                        TextSegment(text="and", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="sad", color=(0, 0, 200), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(
                            text="feelings", color=(50, 50, 50), space_after=True
                        ),
                        TextSegment(text=",", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="and", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="a", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                    ]
                ),
                TextLine(
                    [
                        TextSegment(
                            text="verylongwordthatneedshyphenation",
                            color=(50, 50, 50),
                            space_after=True,
                        ),
                        TextSegment(text=".", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="12345", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text="and", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                    ]
                ),
                TextLine(
                    [
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="heart", color=(50, 50, 50), space_after=True),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="!", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="fire", color=(50, 50, 50), space_after=True),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text=" ", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(50, 50, 50), space_after=True),
                        TextSegment(text="star", color=(50, 50, 50), space_after=True),
                        TextSegment(text=":", color=(50, 50, 50), space_after=False),
                        TextSegment(text=":", color=(0, 0, 0), space_after=True),
                        TextSegment(text="heart", color=(0, 0, 0), space_after=True),
                        TextSegment(text=":", color=(0, 0, 0), space_after=True),
                        TextSegment(text="!", color=(0, 0, 0), space_after=True),
                        TextSegment(text=" ", color=(0, 0, 0), space_after=False),
                        TextSegment(text=":", color=(0, 0, 0), space_after=True),
                        TextSegment(text="fire", color=(0, 0, 0), space_after=True),
                        TextSegment(text=":", color=(0, 0, 0), space_after=True),
                        TextSegment(text=" ", color=(0, 0, 0), space_after=False),
                        TextSegment(text=":", color=(0, 0, 0), space_after=True),
                        TextSegment(text="star", color=(0, 0, 0), space_after=True),
                        TextSegment(text=":", color=(0, 0, 0), space_after=False),
                    ]
                ),
            ],
        ),
    ]

    # Test case 12: Test with custom colors and emphasis
    raw_text12 = (
        "This is a test with <emphasis>custom</emphasis> colors and <link>links</link>."
    )
    personalities12 = {"tags": {"emphasis": "emphasis", "link": "link"}}
    expected_output12 = [
        TextLine(
            [
                TextSegment(text="This", color=(0, 0, 0), space_after=True),
                TextSegment(text=" ", color=(0, 0, 0), space_after=False),
                TextSegment(text="is", color=(0, 0, 0), space_after=True),
                TextSegment(text=" ", color=(0, 0, 0), space_after=False),
                TextSegment(text="a", color=(0, 0, 0), space_after=True),
                TextSegment(text=" ", color=(0, 0, 0), space_after=False),
                TextSegment(text="test", color=(0, 0, 0), space_after=True),
                TextSegment(text=" ", color=(0, 0, 0), space_after=False),
                TextSegment(text="with", color=(0, 0, 0), space_after=True),
                TextSegment(text=" ", color=(0, 0, 0), space_after=False),
                TextSegment(text="custom", color=(255, 255, 0), space_after=True),
                TextSegment(text=" ", color=(0, 0, 0), space_after=False),
                TextSegment(text="colors", color=(0, 0, 0), space_after=True),
                TextSegment(text=" ", color=(0, 0, 0), space_after=False),
                TextSegment(text="and", color=(0, 0, 0), space_after=True),
                TextSegment(text=" ", color=(0, 0, 0), space_after=False),
                TextSegment(text="links", color=(0, 255, 255), space_after=True),
                TextSegment(text=".", color=(0, 0, 0), space_after=False),
            ]
        )
    ]
    run_test_case(
        "Test Case 12 - Custom Colors and Emphasis",
        raw_text12,
        personalities=personalities12,
        expected_output=expected_output12,
    )

    # Test case 13: Test with very long text to test scrolling
    long_text = "This is a very long text to test the scrolling functionality. " * 20
    run_test_case("Test Case 13 - Long Text for Scrolling", long_text)

    # Example of rendering to a surface
    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill((255, 255, 255))
    text_processor.render_lines(screen, 20)
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()
