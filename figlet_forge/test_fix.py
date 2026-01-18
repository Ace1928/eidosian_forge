#!/usr/bin/env python3
"""
Test script to verify the color module imports correctly.
"""

from figlet_forge.color import colored_format

print("Successfully imported all color module components")

# Try creating a colored string
colored_text = colored_format("This is a test", "RED:BLUE")
print(colored_text)

print("Test completed successfully")
