import sys
from typing import List, Tuple

# This module is designed to print a list of unique UTF block and pipe characters
# in various colors covering a spectrum including the rainbow colors, white, black,
# and multiple shades of grey and each color. This enhances the visual representation
# and understanding of these characters in different color contexts.
# Module Header:
# This module is designed to programmatically generate a list of unique UTF block, pipe, and shape characters.
# It covers a comprehensive range of characters by systematically iterating through Unicode code points,
# ensuring no redundancy and a complete coverage of the desired character types.
# This approach enhances maintainability, readability, and future-proofing of the character list.

# Importing necessary modules for Unicode character handling
from typing import List


# Define the function to programmatically generate characters
def generate_utf_characters() -> List[str]:
    """
    Generates a list of UTF block, pipe, and shape characters by systematically iterating through Unicode code points.

    Returns:
        List[str]: A list of unique UTF characters including block, pipe, and shape characters.
    """
    characters: List[str] = []  # Initialize an empty list to store the characters

    # Define the ranges of Unicode code points for block, pipe, and shape characters
    # These ranges were determined based on the Unicode standard documentation
    # and include the most commonly used block drawing and box drawing characters.
    unicode_ranges: List[Tuple[int, int]] = [
        (0x2500, 0x257F),  # Box Drawing
        (0x2580, 0x259F),  # Block Elements
        (0x25A0, 0x25FF),  # Geometric Shapes
    ]

    # Iterate through the defined Unicode ranges
    for start, end in unicode_ranges:
        for code_point in range(start, end + 1):
            try:
                character = chr(code_point)  # Convert code point to character
                characters.append(character)  # Add character to the list
            except ValueError:
                # Log any ValueError exceptions encountered during character conversion
                # This may occur if a code point is not valid or not representable as a character
                print(
                    f"Skipping invalid or non-representable code point: {code_point}",
                    file=sys.stderr,
                )

    return characters  # Return the list of generated characters


# Generate the characters using the defined function
characters: List[str] = generate_utf_characters()

# Comprehensive Spectrum Representation
# This section meticulously defines a comprehensive list of colors, systematically progressing through the spectrum.
# It starts from absolute black, moves through every conceivable shade of grey, transitions through the entire color spectrum
# (red, orange, yellow, green, blue, indigo, violet), and culminates at pure white. This granular approach ensures the highest fidelity
# in representing every possible color and shade, facilitating a vivid and detailed visual representation.
# Define a comprehensive list of colors to represent the full spectrum
# The list includes shades of grey and all colors, meticulously progressing from black through every conceivable shade of grey,
# then through the entire color spectrum from red, orange, yellow, green, blue, indigo, violet, and finally to white,
# with the highest granularity possible.
colors: List[Tuple[int, int, int]] = (
    [
        (0, 0, 0),
    ]  # Absolute Black
    + [
        (i, i, i) for i in range(1, 256)
    ]  # Incrementally increasing shades of grey, from the darkest to the lightest, ensuring a smooth gradient
    + [
        (255, i, 0) for i in range(0, 256)
    ]  # Detailed Red to Orange spectrum, capturing the subtle transition with fine granularity
    + [
        (255, 255, i) for i in range(0, 256)
    ]  # Detailed Orange to Yellow spectrum, capturing every subtle shade in between
    + [
        (255 - i, 255, 0) for i in range(0, 256)
    ]  # Detailed Yellow to Green spectrum, ensuring every shade is represented
    + [
        (0, 255, i) for i in range(0, 256)
    ]  # Detailed Green to Blue spectrum, capturing the full range of shades in between
    + [
        (0, 255 - i, 255) for i in range(0, 256)
    ]  # Detailed Blue to Indigo spectrum, with fine granularity to capture the transition
    + [
        (i, 0, 255) for i in range(0, 256)
    ]  # Detailed Indigo to Violet spectrum, ensuring a smooth gradient
    + [
        (255, i, 255) for i in range(0, 256)
    ]  # Detailed Violet to White spectrum, capturing every possible shade in between
    + [
        (255, 255, 255),  # Pure White
    ]
)

# This comprehensive approach ensures that every possible color and shade from black to white is represented with the highest fidelity,
# allowing for a vivid and detailed visual representation of characters in the spectrum.


# This comprehensive approach ensures that every possible color and shade from black to white is represented with the highest fidelity,
# allowing for a vivid and detailed visual representation of characters in the spectrum.
def print_characters_in_colors(
    characters: List[str], colors: List[Tuple[int, int, int]]
) -> None:
    """
    Prints each character in the provided list in a series of colors.

    :param characters: A list of characters to be printed.
    :param colors: A list of RGB color tuples.
    """
    for char in characters:
        print(f"Character: {char} - Unicode: {ord(char)}", end=" | ")
        for color in colors:
            # ANSI escape code for color formatting
            print(f"\033[38;2;{color[0]};{color[1]};{color[2]}m{char}\033[0m", end=" ")
        print()  # Newline after printing all colors for a character


# Execute the function to print characters in colors
print_characters_in_colors(characters, colors)
