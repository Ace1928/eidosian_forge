#!/bin/bash
# 🌀 Eidosian Directory Structure Creator
# Creates the perfect documentation directory structure with elegant precision

# Set up colors for beautiful terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Creating Eidosian Documentation Directory Structure ║${NC}"
echo -e "${BLUE}╚═════════════════════════════════════════════════════╝${NC}"

# Determine the repo root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$REPO_ROOT/docs"

# Create primary documentation directories
mkdir -p "$DOCS_DIR"
mkdir -p "$DOCS_DIR/_build/html"
mkdir -p "$DOCS_DIR/_static/css"
mkdir -p "$DOCS_DIR/_static/js"
mkdir -p "$DOCS_DIR/_static/img"
mkdir -p "$DOCS_DIR/_templates"
mkdir -p "$DOCS_DIR/user_docs"
mkdir -p "$DOCS_DIR/auto_docs"
mkdir -p "$DOCS_DIR/ai_docs"
mkdir -p "$DOCS_DIR/assets"

echo -e "${GREEN}✓${NC} Created documentation directory structure"

# Create minimal CSS if it doesn't exist
CSS_FILE="$DOCS_DIR/_static/css/custom.css"
if [ ! -f "$CSS_FILE" ]; then
    cat > "$CSS_FILE" << EOF
/* Custom CSS for documentation */
:root {
    --font-size-base: 16px;
    --line-height-base: 1.5;
}

.wy-nav-content {
    max-width: 900px;
}

.highlight {
    border-radius: 4px;
}

/* Dark mode support for code blocks */
@media (prefers-color-scheme: dark) {
    .highlight {
        background-color: #2d2d2d;
    }
}
EOF
    echo -e "${GREEN}✓${NC} Created custom CSS file"
fi

# Create minimal JavaScript if it doesn't exist
JS_FILE="$DOCS_DIR/_static/js/custom.js"
if [ ! -f "$JS_FILE" ]; then
    cat > "$JS_FILE" << EOF
/* Custom JavaScript for documentation */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Doc Forge documentation loaded');
});
EOF
    echo -e "${GREEN}✓${NC} Created custom JavaScript file"
fi

# Create placeholder README files in empty directories
for DIR in "$DOCS_DIR/user_docs" "$DOCS_DIR/auto_docs" "$DOCS_DIR/api_docs" "$DOCS_DIR/assets"; do
    if [ -z "$(ls -A "$DIR")" ]; then
        echo "# $(basename "$DIR")" > "$DIR/README.md"
        echo -e "${GREEN}✓${NC} Created placeholder README in $(basename "$DIR")"
    fi
done

echo -e "\n${GREEN}Documentation directory structure created successfully!${NC}"
exit 0
