#!/bin/bash

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ Figlet Forge Font Tester                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# This script tests figlet_forge with various fonts and display options
# to verify rendering functionality.

PYTHON_CMD="python3"
TEXT="Hello World"
COLOR_OPTION=""
UNICODE_OPTION=""

usage() {
    echo "Usage: $0 [options]"
    echo "  -f FONT          Specify font to test (default: test all fonts)"
    echo "  -t TEXT          Specify text to render (default: \"$TEXT\")"
    echo "  -c               Enable color output testing"
    echo "  -u               Enable Unicode output testing"
    echo "  -p PYTHON_CMD    Python command (default: $PYTHON_CMD)"
    echo "  -h               Show this help"
}

test_font() {
    local font=$1
    echo "==== Testing font: $font ===="

    cmd="$PYTHON_CMD -c \"from figlet_forge import Figlet, print_figlet; print_figlet('$TEXT', font='$font'$COLOR_OPTION)\""
    echo "Command: $cmd"
    eval "$cmd"
    echo ""
}

test_all_fonts() {
    echo "Getting font list..."
    FONTS=$($PYTHON_CMD -c "from figlet_forge import Figlet; print(' '.join(sorted(Figlet().getFonts())))")

    for font in $FONTS; do
        test_font "$font"
    done
}

# Parse options
while getopts "f:t:cup:h" opt; do
    case $opt in
        f) FONT="$OPTARG" ;;
        t) TEXT="$OPTARG" ;;
        c) COLOR_OPTION=", colors='BLUE:'" ;;
        u) UNICODE_OPTION=", unicode_aware=True" ;;
        p) PYTHON_CMD="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

# Ensure the package can be imported
if ! $PYTHON_CMD -c "import figlet_forge" 2>/dev/null; then
    echo "ERROR: Cannot import figlet_forge. Is it installed?" >&2
    exit 1
fi

# Run tests
if [ -n "$FONT" ]; then
    test_font "$FONT"
else
    test_all_fonts
fi

echo "Font testing complete!"
