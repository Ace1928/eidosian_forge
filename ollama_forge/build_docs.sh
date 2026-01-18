#!/usr/bin/env bash
# Build documentation for ollama_forge
# ✨ Eidosian Refinement: Concise, Precise, Structured Flow ⚙️
# Humor included to lighten tension, preserving clarity and momentum 💥

set -e  # End immediately on any error

# 🌈 Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

DOCS_DIR="docs"
BUILD_DIR="${DOCS_DIR}/_build/html"

echo -e "${BLUE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Ollama Forge Documentation Builder ⚡🚀       ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════╝${NC}"

# 🛡️ Error handling with clarity and precision
error_handler() {
    echo -e "${RED}⚠️  Error at line $1.${NC}"
    echo -e "${RED}Build process halted.${NC}"
    exit 1
}

trap 'error_handler $LINENO' ERR

# Move to script directory 🌐
cd "$(dirname "$0")" || {
    echo -e "${RED}Unable to enter script directory!${NC}"
    exit 1
}

echo -e "${YELLOW}Current working directory: $(pwd)${NC}"

# Ensure existence of docs directory 📁
if [ ! -d "$DOCS_DIR" ]; then
    echo -e "${YELLOW}🌀 Creating docs directory...${NC}"
    mkdir -p "$DOCS_DIR"
    echo -e "${GREEN}✓ docs directory ready${NC}"
fi

# Check for docs/requirements.txt 📑
if [ ! -f "${DOCS_DIR}/requirements.txt" ]; then
    echo -e "${YELLOW}🌀 Creating default requirements.txt...${NC}"
    cat > "${DOCS_DIR}/requirements.txt" << EOF
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
myst-parser>=0.15.0
sphinx-autobuild>=0.7.1
sphinx-copybutton>=0.3.1
sphinx-autodoc-typehints>=1.11.1
EOF
    echo -e "${GREEN}✓ requirements.txt in place${NC}"
fi

# Clear past builds 💥
echo -e "${BLUE}🧹 Removing previous builds...${NC}"
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}✓ Old build removed${NC}"
else
    echo -e "${GREEN}✓ No old build to remove${NC}"
fi

# Check doc dependencies with minimal friction 🔍
echo -e "${BLUE}🔎 Verifying documentation dependencies...${NC}"
MISSING_DEPS=0
for pkg in sphinx myst-parser sphinx-rtd-theme sphinx-autobuild; do
    if ! pip show "$pkg" &> /dev/null; then
        MISSING_DEPS=1
        break
    fi
done

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${YELLOW}🌀 Installing missing doc dependencies...${NC}"
    pip install -r "${DOCS_DIR}/requirements.txt"
    echo -e "${GREEN}✓ Dependencies ready${NC}"
else
    echo -e "${GREEN}✓ All dependencies present${NC}"
fi

# Minimal conf.py if missing 🔧
if [ ! -f "${DOCS_DIR}/conf.py" ]; then
    echo -e "${YELLOW}⚠️  No conf.py found—creating minimal configuration...${NC}"
    cat > "${DOCS_DIR}/conf.py" << EOF
# Sphinx configuration: Eidosian minimalism 🔄
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Ollama Forge'
copyright = '2025, Lloyd Handyside'
author = 'Lloyd Handyside'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

try:
    from version import get_version_string
    version = get_version_string()
    release = version
except ImportError:
    version = 'unknown'
    release = 'unknown'
EOF
    echo -e "${GREEN}✓ Minimal conf.py created${NC}"
fi

# Minimal index if missing 🎯
if [ ! -f "${DOCS_DIR}/index.rst" ] && [ ! -f "${DOCS_DIR}/index.md" ]; then
    echo -e "${YELLOW}⚠️  No index document—creating index.md...${NC}"
    cat > "${DOCS_DIR}/index.md" << EOF
# Ollama Forge Documentation

🎨 Welcome to Ollama Forge docs! 
Flowing with Eidosian minimalism ⚡

\`\`\`{toctree}
:maxdepth: 2
:caption: Contents:

README
\`\`\`

## Indices and tables

* {ref}\`genindex\`
* {ref}\`modindex\`
* {ref}\`search\`
EOF
    if [ ! -f "${DOCS_DIR}/README.md" ] && [ -f "README.md" ]; then
        ln -sf "../README.md" "${DOCS_DIR}/README.md"
        echo -e "${GREEN}✓ Linked main README to docs${NC}"
    fi
    echo -e "${GREEN}✓ index.md is set${NC}"
fi

# Build documentation 🚧
echo -e "${BLUE}📚 Building documentation...${NC}"
sphinx-build -b html "$DOCS_DIR" "$BUILD_DIR"
echo -e "${GREEN}✓ Build completed${NC}"

# Check final artifact 🏁
if [ -f "${BUILD_DIR}/index.html" ]; then
    echo -e "${GREEN}┌─────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}│    Documentation build succeeded!      │${NC}"
    echo -e "${GREEN}└─────────────────────────────────────────┘${NC}"
    echo -e "📑 Output location: ${BUILD_DIR}/index.html"

    if [ -n "$DISPLAY" ]; then
        echo -e "${BLUE}Attempting to open in default browser...${NC}"
        if command -v xdg-open &> /dev/null; then
            xdg-open "${BUILD_DIR}/index.html"
        elif command -v open &> /dev/null; then
            open "${BUILD_DIR}/index.html"
        else
            echo -e "${YELLOW}Cannot auto-open; open manually:${NC}"
            echo "file://$(realpath "${BUILD_DIR}/index.html")"
        fi
    else
        echo -e "${YELLOW}No graphical display detected. Open manually:${NC}"
        echo "file://$(realpath "${BUILD_DIR}/index.html")"
    fi
else
    echo -e "${RED}Build process did not produce an index file. 🛑${NC}"
    exit 1
fi

# Next steps 🎬
echo -e "\n${BLUE}⚙️ Next Steps:${NC}"
echo "1. Verify generated docs thoroughly."
echo "2. Optionally serve: python -m http.server --directory ${BUILD_DIR}"
echo "3. Publish externally if desired (e.g., Read the Docs)."

exit 0