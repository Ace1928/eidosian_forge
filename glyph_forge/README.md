# 🖼️ Glyph Forge ⚡

> _"Pixels to Symbols. Cinematic ASCII in one command. Defining the Eidosian visual signature."_

## 🧠 Overview

`glyph_forge` is the high-fidelity visual translation engine for the Eidosian ecosystem. It converts raw pixels (images, video, webcams, browser streams) into structurally elegant ASCII/ANSI art with extreme performance. It supports a vast array of color modes, dithering algorithms, and real-time streaming capabilities, serving as the primary source for Eidosian branding and interactive terminal UIs.

```ascii
      ╭───────────────────────────────────────────╮
      │               GLYPH FORGE                 │
      │    < Image | Video | Streaming | TUI >    │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   MODULAR ENGINE    │   │  RICH RENDERERS │
      │ (OpenCV / Pillow)   │   │ (ANSI / SVG / HT)│
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Visual Processing & Rendering
- **Test Coverage**: 420+ tests passing (97%+ logic coverage).
- **Performance Baseline**:
  - Gradient Mode: **74 fps** @ 480p
  - Braille Mode: **144 fps** @ 480p (Ultra-fast monochrome)
- **Architecture**:
  - `transformers/`: Advanced image analysis (edge detection, depth maps).
  - `renderers/`: Output formatters for terminal, web (HTML), and vector (SVG) targets.
  - `streaming/`: High-performance modular engine for real-time video-to-ASCII.
  - `ui/`: Full-featured Textual TUI for interactive conversion.

## 🚀 Usage & Workflows

### Image & Banner Generation

```bash
# Convert an image with maximum fidelity
glyph-forge imagize convert image.png --color truecolor --dither --edge-enhance

# Generate a stylized text banner
glyph-forge bannerize "EIDOS" --font slant --style boxed --color
```

### Real-Time Streaming

```bash
# Stream a YouTube video directly to the terminal
glyph-forge stream https://youtube.com/watch?v=VIDEO_ID

# Capture and render the current screen (Requires X11/Wayland)
glyph-forge stream --screen
```

### Interactive Command Center

Launch the integrated terminal dashboard:
```bash
glyph-forge interactive
```

## 🔗 System Integration

- **Terminal Forge**: Supplies the base themes and color palettes used by the renderers.
- **Agent Forge**: Powers the "Visual Cortex" of agents, allowing them to "see" and summarize terminal state or external media.
- **Article Forge**: Utilized to generate high-fidelity ASCII illustrations for publishable markdown content.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate 50+ fragmented documentation files into a unified standard.
- [x] Stabilize Termux-specific OpenCV/FFmpeg fallback paths.

### Future Vector (Phase 3+)
- Implement "Neural Dithering" using lightweight local models from `llm_forge` to optimize character selection based on semantic content rather than just luminance.

---
*Generated and maintained by Eidos.*
