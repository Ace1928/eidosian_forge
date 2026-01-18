"""
Figlet Font Handler for Figlet Forge.

This module handles loading and parsing FIGlet font files, which define
the ASCII art representations of characters used by the FIGlet system.
"""

import logging
import os
import re
import zipfile
from importlib import resources
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Set,
    TypedDict,
    Union,
)

from ..core.exceptions import FontNotFound

# Configure logger for the font module
logger = logging.getLogger(__name__)

# Define type aliases for better clarity and precision
CharacterMap = Dict[str, List[str]]
WidthMap = Dict[str, int]


class FontInfo(TypedDict, total=False):
    """TypedDict for font metadata."""

    name: str
    author: str
    date: str
    version: str
    comment: str

    # Optional fields
    magic_number: Optional[int]
    full_width: Optional[int]


class FigletFont:
    """
    Represents a FIGlet font with character mapping and metadata.

    This class loads and parses FIGlet font files, handling the specifics
    of different font formats and providing methods to access character
    representations and metadata.
    """

    def __init__(self) -> None:
        """Initialize an empty FIGlet font."""
        self.comment = ""
        self.chars: CharacterMap = {}
        self.width: WidthMap = {}
        self.font_name = ""
        self.height = 0
        self.base_line = 0
        self.max_length = 0
        self.old_layout = 0
        self.print_direction = 0
        self.full_width = 0
        self.code_tags_depth = 0
        self.hard_blank = ""

        # Additional metadata for enhanced font tracking
        self.loaded_from: Optional[str] = None
        self.author: Optional[str] = None
        self.version: Optional[str] = None
        self.date: Optional[str] = None

        # Additional attributes for compatibility
        self.info: str = ""
        self.info_dict: Dict[str, str] = {}

    def load_font(
        self,
        font_path: Optional[Union[str, Path]] = None,
        font_name: Optional[str] = None,
    ) -> bool:
        """
        Load a font from file or embedded resources.

        Args:
            font_path: Path to the font file, if None uses font_name
            font_name: Name of the embedded font to load

        Returns:
            True if font was loaded successfully, False otherwise
        """
        success = False

        # If no specific font is requested, use the default
        if font_path is None and font_name is None:
            font_name = "standard"

        # Try direct file load first (if path provided)
        if font_path is not None:
            try:
                # Ensure path is a string for file operations
                path_str = str(font_path)

                if os.path.isfile(path_str):
                    with open(path_str, "rb") as font_file:
                        data = font_file.read().decode("latin-1", errors="replace")
                        success = self.parse_font(data)
                        if success:
                            self.loaded_from = path_str
                            self.font_name = os.path.basename(path_str).split(".")[0]
            except (OSError, UnicodeDecodeError, RuntimeError) as e:
                # Log the error but continue to try other methods
                logger.warning(f"Error loading font file: {e}")

        # If direct file load failed and we have a font name, try package fonts
        if not success and font_name is not None:
            # First try built-in package fonts - search in priority order
            font_package_paths = [
                "figlet_forge.fonts",  # Base fonts
                "figlet_forge.fonts.standard",  # Standard fonts
                "figlet_forge.fonts.contrib",  # Contrib fonts
                "figlet_forge.data.fonts",  # Legacy package path
            ]

            for package_path in font_package_paths:
                try:
                    # Try both .flf and .tlf extensions
                    for ext in [".flf", ".tlf"]:
                        resource_name = f"{font_name}{ext}"
                        try:
                            font_data = resources.read_binary(
                                package_path, resource_name
                            )
                            success = self.parse_font(
                                font_data.decode("latin-1", errors="replace")
                            )
                            if success:
                                self.loaded_from = (
                                    f"resource:{package_path}:{font_name}"
                                )
                                self.font_name = font_name
                                return True
                        except (FileNotFoundError, Exception) as resource_error:
                            # Generalized exception handling for resource errors
                            # This covers ResourceReadError without explicitly importing it
                            logger.debug(
                                f"Font {font_name}{ext} not found in {package_path}: {resource_error}"
                            )
                except (ImportError, ModuleNotFoundError) as e:
                    # Log but continue to next package path
                    logger.debug(f"Package {package_path} not found: {e}")

        # If we still don't have a font, try the system font paths
        if not success and font_name is not None:
            success = self._find_font_in_paths(font_name)

        # If everything failed, try to load the default "standard" font
        if not success and font_name != "standard":
            logger.info(
                f"Failed to load font '{font_name}', falling back to 'standard'"
            )
            return self.load_font(None, "standard")

        return success

    def _find_font_in_paths(self, font_name: str) -> bool:
        """
        Search for the font in known system paths.

        Args:
            font_name: Name of the font to find

        Returns:
            True if found and loaded successfully, False otherwise
        """
        # Standard paths where figlet fonts may be installed
        search_paths = [
            "/usr/share/figlet",  # Common Linux location
            "/usr/local/share/figlet",  # Common BSD/macOS location
            "/usr/share/figlet/fonts",  # Alternative location
            "/usr/local/lib/figlet/fonts",
            os.path.expanduser("~/.figlet/fonts"),  # User-specific location
            os.path.expanduser("~/figlet/fonts"),  # Alternative user location
        ]

        # Add current directory and script directory as final fallbacks
        search_paths.append(os.path.dirname(os.path.abspath(__file__)))
        search_paths.append(os.getcwd())

        # Search for the font file
        for path in search_paths:
            if not os.path.exists(path):
                continue

            # Find all font files in directory
            found_files: List[str] = []
            for root, _, files in os.walk(path):
                for file in files:
                    if file.startswith(f"{font_name}.") or file == font_name:
                        found_files.append(os.path.join(root, file))

            # Try loading each candidate font file
            for file_path in found_files:
                try:
                    with open(file_path, "rb") as font_file:
                        data = font_file.read().decode("latin-1", errors="replace")
                        if self.parse_font(data):
                            self.loaded_from = file_path
                            self.font_name = font_name
                            return True
                except (OSError, UnicodeDecodeError) as e:
                    # Log and try next file
                    logger.debug(f"Error trying font file {file_path}: {e}")

        return False

    def parse_font(self, data: str) -> bool:
        """
        Parse font data and load character definitions.

        Args:
            data: The raw font data as a string

        Returns:
            True if parsing succeeded, False otherwise

        Raises:
            RuntimeError: If the font format is invalid
        """
        lines = data.splitlines()

        # Need at least the header line
        if not lines:
            return False

        # Parse the header and extract basic font properties
        try:
            header = lines[0].split()
            if len(header) < 6:
                raise RuntimeError("Invalid header format")

            # Extract header fields
            self.hard_blank = header[0][-1]
            self.height = int(header[1])
            self.base_line = int(header[2])
            self.max_length = int(header[3])
            self.old_layout = int(header[4])
            # Attempt to handle both old and new header formats
            self.print_direction = 0
            self.full_width = 0
            self.code_tags_depth = 0

            if len(header) > 5:
                comment_lines = int(header[5])
                self.comment = "\n".join(lines[1 : comment_lines + 1])
                # Extract metadata from comment
                self._extract_metadata_from_comment()
                current_line = comment_lines + 1
            else:
                current_line = 1
        except (ValueError, IndexError) as e:
            raise RuntimeError(f"Error parsing font header: {e}") from e

        # We now have a font with header, but no character data yet
        # Create empty character data for required characters
        self._create_blank_characters()

        # Process character definitions
        try:
            # Detect font format more accurately
            format_type = self._detect_font_format(lines)

            if format_type == "german":
                success = self._load_german_format(lines, current_line)
                if not success:
                    # Fallback to standard format if German format fails
                    return self._load_standard_format(lines, current_line)
            elif format_type == "codetagged":
                success = self._load_codetagged_format(lines, current_line)
                if not success:
                    # Fallback to standard format if codetagged format fails
                    return self._load_standard_format(lines, current_line)
            else:  # Standard format
                return self._load_standard_format(lines, current_line)

            return True

        except Exception as e:
            # If an error occurs, ensure we leave a clean state
            self.chars = {}
            self.width = {}
            logger.error(f"Error parsing font data: {e}")
            return False

    def _detect_font_format(self, lines: List[str]) -> str:
        """
        Detect the format of a font file by analyzing its content structure.

        Args:
            lines: Font file content as list of lines

        Returns:
            Format type: 'standard', 'german', or 'codetagged'
        """
        # Join all lines for efficient searching
        content = "\n".join(lines)

        # Check for German format marker - exact match only
        if "\n@\n" in content:
            # Verify it's not just a random occurrence by checking if it appears
            # after the header section (rough heuristic)
            header_end = content.find("\n\n")
            if header_end > 0 and content.find("\n@\n", header_end) > 0:
                return "german"

        # Check for codetagged format marker - exact match only
        if "\n@+@\n" in content:
            # Similar verification
            header_end = content.find("\n\n")
            if header_end > 0 and content.find("\n@+@\n", header_end) > 0:
                return "codetagged"

        # Default to standard format
        return "standard"

    def _load_standard_format(self, lines: List[str], current_line: int) -> bool:
        """
        Load a standard format FIGlet font.

        Args:
            lines: Font file content as list of lines
            current_line: Starting line number for parsing

        Returns:
            True if parsing succeeded, False otherwise
        """
        try:
            # Parse each character block
            while current_line < len(lines):
                line_count = 0
                char_lines: List[str] = []

                # Get the character definition lines
                while line_count < self.height and current_line + line_count < len(
                    lines
                ):
                    line = lines[current_line + line_count]
                    # Handle lines that might be too short (missing endmark)
                    if line:
                        char_lines.append(line[:-1])  # Remove endmark
                    else:
                        char_lines.append("")
                    line_count += 1

                if not char_lines:
                    current_line += max(1, line_count)  # Skip to next potential block
                    continue

                # Determine which character this is
                if (
                    current_line + self.height <= len(lines)
                    and lines[current_line + self.height - 1]
                ):
                    # Get the last character of the last line in this block
                    last_line = lines[current_line + self.height - 1]
                    if last_line:
                        curr_char = last_line[-1]
                    else:
                        curr_char = " "  # Default to space for empty lines
                else:
                    curr_char = " "  # Default to space if out of bounds

                # Store the character, removing trailing spaces for width calculation
                width = (
                    max(
                        len(line.rstrip("\r\n" + self.hard_blank))
                        for line in char_lines
                    )
                    if char_lines
                    else 0
                )

                self.chars[curr_char] = char_lines
                self.width[curr_char] = width

                current_line += self.height

            return bool(self.chars)  # Success if we have at least some characters

        except Exception as e:
            logger.debug(f"Error in standard format parsing: {e}")
            return False

    def _load_german_format(self, lines: List[str], current_line: int) -> bool:
        """
        Load a German format FIGlet font with improved error handling.

        Args:
            lines: Font file content as list of lines
            current_line: Starting line number for parsing

        Returns:
            True if parsing succeeded, False otherwise
        """
        try:
            # German font format parsing
            self.chars = {}
            self.width = {}

            # Find the marker line more safely
            marker_line = -1
            for i in range(current_line, len(lines)):
                if lines[i] == "@":  # Both exact matches possible
                    marker_line = i
                    break

            if marker_line < 0:
                # Try again with newline
                for i in range(current_line, len(lines)):
                    if lines[i] == "@\n" or lines[i] == "@\r\n":
                        marker_line = i
                        break

            if marker_line < 0:
                # No German format marker found at all
                logger.debug("Cannot find marker line in German font format")
                return False

            # Skip over marker line
            current_line = marker_line + 1

            # Process character definitions
            chr_code = ord(" ")  # Start with space

            for _ in range(0, 95):  # 95 printable ASCII chars
                if current_line + self.height > len(lines):
                    # Not enough lines left in the file
                    return bool(self.chars)  # Return success if we got some chars

                char_lines: List[str] = []
                char = chr(chr_code)

                # Read each line of the character
                for h in range(self.height):
                    if current_line + h < len(lines):
                        char_lines.append(lines[current_line + h].rstrip("\r\n"))
                    else:
                        # Pad with empty lines if data is missing
                        char_lines.append("")

                # Store the character data
                width = (
                    max(len(line.rstrip()) for line in char_lines) if char_lines else 0
                )
                self.chars[char] = char_lines
                self.width[char] = width

                chr_code += 1
                current_line += self.height

            return True

        except Exception as e:
            logger.debug(f"Error in German format parsing: {e}")
            return False

    def _load_codetagged_format(self, lines: List[str], current_line: int) -> bool:
        """
        Load a codetagged format FIGlet font with improved error handling.

        Args:
            lines: Font file content as list of lines
            current_line: Starting line number for parsing

        Returns:
            True if parsing succeeded, False otherwise
        """
        try:
            # Codetagged font format parsing
            self.chars = {}
            self.width = {}

            # Find the code-tagged section safely
            marker_line = -1
            for i in range(current_line, len(lines)):
                if lines[i] in ("@+@", "@+@\n", "@+@\r\n"):
                    marker_line = i
                    break

            if marker_line < 0:
                logger.debug("Cannot find marker line in codetagged font format")
                return False

            # Skip over marker line
            current_line = marker_line + 1

            # Process character definitions until we reach the end marker
            while current_line < len(lines):
                if lines[current_line] in ("@@", "@@\n", "@@\r\n"):
                    break  # End of codetagged section

                # Get character code from the first line
                if not (
                    current_line < len(lines) and lines[current_line].startswith("+")
                ):
                    current_line += 1
                    continue

                try:
                    # Parse character code, handling potential formatting
                    code_line = lines[current_line].strip()
                    char_code = int(code_line[1:])
                    current_line += 1
                    char = chr(char_code)

                    # Get character definition lines
                    char_lines: List[str] = []
                    for _ in range(self.height):
                        if current_line < len(lines) and not lines[
                            current_line
                        ].startswith("+"):
                            line = lines[current_line].rstrip("\r\n")
                            char_lines.append(line)
                            current_line += 1
                        else:
                            # Pad with empty lines if data is missing
                            char_lines.append("")

                    # Store character data
                    width = (
                        max(len(line.rstrip()) for line in char_lines)
                        if char_lines
                        else 0
                    )
                    self.chars[char] = char_lines
                    self.width[char] = width

                    # Skip end marker for this character
                    if current_line < len(lines) and lines[current_line] in (
                        "+@",
                        "+@\n",
                        "+@\r\n",
                    ):
                        current_line += 1

                except (ValueError, IndexError) as e:
                    # Log and skip invalid character definitions
                    logger.debug(f"Invalid character definition: {e}")
                    current_line += 1

            return bool(self.chars)  # Success if we parsed some characters

        except Exception as e:
            logger.debug(f"Error in codetagged format parsing: {e}")
            return False

    def _create_blank_characters(self) -> None:
        """Create blank representation for basic characters."""
        # Create an empty line template
        empty_line = " " * self.max_length

        # Initialize the dictionary with empty characters
        self.chars = {}
        self.width = {}

        # Set at minimum the printable ASCII chars
        for i in range(32, 127):
            char = chr(i)
            self.chars[char] = [empty_line] * self.height
            self.width[char] = self.max_length

    def get_character(self, char: str) -> List[str]:
        """
        Get the ASCII art for a specific character.

        Args:
            char: The character to retrieve

        Returns:
            List of strings representing the ASCII art for the character
        """
        if not char or len(char) == 0:
            return self.chars.get(" ", [" " * self.max_length] * self.height)

        # For multi-character strings, just return the first character
        if len(char) > 1:
            char = char[0]

        # Return the character data, or space if not found
        if char in self.chars:
            return self.chars[char]
        else:
            # Try to handle special characters
            try:
                # Check if it's a unicode character we can map
                for fallback in ["?", " "]:
                    if fallback in self.chars:
                        return self.chars[fallback]
            except Exception as e:
                # Log instead of silent pass for better error handling
                logger.debug(f"Error handling special character '{char}': {e}")

            # Default fallback
            return [" " * self.max_length] * self.height

    def get_width(self, char: str) -> int:
        """
        Get the width of a specific character.

        Args:
            char: The character to check

        Returns:
            Width of the character in columns
        """
        if not char or len(char) == 0:
            return self.width.get(" ", self.max_length)

        # For multi-character strings, just get the first character
        if len(char) > 1:
            char = char[0]

        # Return the character width, or default if not found
        if char in self.width:
            return self.width[char]
        else:
            # Try to use fallback characters
            for fallback in ["?", " "]:
                if fallback in self.width:
                    return self.width[fallback]
            # Last resort
            return self.max_length

    def _extract_metadata_from_comment(self) -> None:
        """Extract metadata fields from font comment."""
        if not self.comment:
            return

        # Extract common metadata fields
        metadata_patterns = {
            "name": r"(?:Name|Font)[:\s]+([^\n]+)",
            "author": r"(?:Author|By)[:\s]+([^\n]+)",
            "date": r"(?:Date|Created)[:\s]+([^\n]+)",
            "version": r"(?:Version)[:\s]+([^\n]+)",
        }

        for field, pattern in metadata_patterns.items():
            match = re.search(pattern, self.comment, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                setattr(self, field, value)

    @classmethod
    def get_fonts(cls) -> List[str]:
        """
        Get a list of all available font names.

        Returns:
            List of available font names in alphabetical order
        """
        fonts: Set[str] = set()

        # Check built-in package fonts - precisely organized by priority
        package_paths = [
            "figlet_forge.fonts",  # Base fonts
            "figlet_forge.fonts.standard",  # Standard fonts
            "figlet_forge.fonts.contrib",  # Contrib fonts
            "figlet_forge.data.fonts",  # Legacy package path
        ]

        for package_path in package_paths:
            try:
                # Use importlib.resources for Python 3.7+ compatibility
                for resource in resources.contents(package_path):
                    if resource.endswith((".flf", ".tlf")):
                        font_name = resource.rsplit(".", 1)[0]  # Remove extension
                        fonts.add(font_name)
            except (ImportError, ModuleNotFoundError) as e:
                # Log and continue with other sources
                logger.debug(f"Error accessing package resources {package_path}: {e}")

        # Check system font directories
        font_dirs = [
            "/usr/share/figlet",
            "/usr/local/share/figlet",
            "/usr/share/figlet/fonts",
            os.path.expanduser("~/.figlet/fonts"),
        ]

        for directory in font_dirs:
            try:
                if os.path.isdir(directory):
                    for file in os.listdir(directory):
                        if file.endswith((".flf", ".tlf")):
                            font_name = file.rsplit(".", 1)[0]  # Remove extension
                            fonts.add(font_name)
            except OSError as e:
                logger.debug(f"Error accessing font directory {directory}: {e}")

        # Ensure we always have at least the standard font
        fonts.add("standard")

        # Convert to sorted list for consistent output
        return sorted(list(fonts))

    def info_font(self, font_name: Optional[str] = None) -> Optional[FontInfo]:
        """
        Get information about a specific font.

        Args:
            font_name: Name of the font to query

        Returns:
            Dictionary with font information or None if font not found
        """
        # If no font_name provided, use current font
        if font_name is None and self.font_name:
            # Return info about currently loaded font
            return {
                "name": self.font_name,
                "author": self.author or "Unknown",
                "date": self.date or "Unknown",
                "version": self.version or "Unknown",
                "comment": self.comment or "",
                "magic_number": 0,  # Default value for compatibility
                "full_width": self.full_width,
            }

        # Load a temporary font to get its info
        temp_font = FigletFont()
        if temp_font.load_font(font_name=font_name):
            return {
                "name": temp_font.font_name,
                "author": temp_font.author or "Unknown",
                "date": temp_font.date or "Unknown",
                "version": temp_font.version or "Unknown",
                "comment": temp_font.comment or "",
                "magic_number": 0,  # Default value for compatibility
                "full_width": temp_font.full_width,
            }

        return None

    def install_fonts(self, zip_path: Union[str, Path]) -> List[str]:
        """
        Install fonts from a zip file to user fonts directory.

        Args:
            zip_path: Path to zip file containing fonts

        Returns:
            List of installed font names
        """
        installed_fonts: List[str] = []

        # Create user fonts directory if it doesn't exist
        user_fonts_dir = os.path.expanduser("~/.figlet/fonts")
        os.makedirs(user_fonts_dir, exist_ok=True)

        try:
            # Ensure zip_path is a string for zipfile operations
            zip_path_str = str(zip_path)

            with zipfile.ZipFile(zip_path_str, "r") as z:
                # Extract only .flf and .tlf files
                for file_info in z.infolist():
                    if file_info.filename.endswith((".flf", ".tlf")):
                        # Extract the file to user fonts directory
                        z.extract(file_info, user_fonts_dir)
                        # Get font name by removing extension
                        font_name = os.path.basename(file_info.filename).rsplit(".", 1)[
                            0
                        ]
                        installed_fonts.append(font_name)

            return installed_fonts
        except (zipfile.BadZipFile, OSError) as e:
            raise FontNotFound(f"Failed to install fonts: {e}") from e

    # Methods for backward compatibility with older tests
    # Using proper noqa comments to suppress linter warnings for camelCase names
    def getCharacter(self, char: str) -> List[str]:  # noqa: N802
        """Backward compatibility method for get_character."""
        return self.get_character(char)

    def getWidth(self, char: str) -> int:  # noqa: N802
        """Backward compatibility method for get_width."""
        return self.get_width(char)

    @classmethod
    def getFonts(cls) -> List[str]:  # noqa: N802
        """Backward compatibility method for get_fonts."""
        return cls.get_fonts()
