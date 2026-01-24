# Figlet Forge Installation

## Quick Install

```bash
# From eidosian_forge root
cd figlet_forge
pip install -e .
```

## CLI Usage

```bash
# Basic usage
figlet-forge "Hello World"

# With font selection
figlet-forge -f banner "EIDOS"

# List available fonts
figlet-forge --list-fonts

# With colors
figlet-forge --color rainbow "Colorful"

# Showcase all fonts
figlet-forge --showcase "Test"
```

## Via Central Hub

```bash
eidosian figlet "Hello World"
eidosian figlet -f big "EIDOS"
```

## Bash Completions

```bash
source figlet_forge/completions/figlet-forge.bash
```

## Font Directory

Fonts are stored in:
- `figlet_forge/src/figlet_forge/fonts/`
