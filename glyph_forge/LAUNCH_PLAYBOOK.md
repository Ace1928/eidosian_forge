# Glyph Forge — Launch & Viral Growth Playbook

## 0) Goal

Make Glyph Forge the default answer to: "How do I make my terminal look like art?"

## 1) Launch Stages

### Stage A — Pre‑Launch (1–2 weeks)

- **Package polish**:
  - Confirm `pip install -e .` works cleanly.
  - Add a `--demo` mode that ships a small built‑in sample.
  - Provide a single "wow" command.
- **Content seed**:
  - 3 short clips: image, video, webcam.
  - 6 static images in `examples/` or `assets/`.
- **Community prep**:
  - Draft 4 post variants for each platform.
  - Create a short thread narrative: “terminal → cinema”.

### Stage B — Launch (3–7 days)

- **Day 1**: Announce + short demo clip.
- **Day 2**: Technical breakdown (speed, braille renderer).
- **Day 3**: Community prompt: "Show us your terminal art."
- **Day 4–7**: Highlight user submissions daily.

### Stage C — Post‑Launch (ongoing)

- Release tiny updates weekly.
- Share one new clip per update.
- Build community gallery.

## 2) Viral Loops

### Loop 1 — Output‑to‑Share
User runs a command → output is beautiful → exported to share.

Product requirements:
- `--share svg|html|png` exports automatically.
- `--open` can open output in browser.
- `--copy` copies banner art to clipboard.

### Loop 2 — Remix & Challenge
Prompt users to remix output; feature on README/Gallery.

Product requirements:
- `glyph-forge remix <file>` to apply styles/gradients to existing render.
- `--seed` for reproducible random styles.

### Loop 3 — Stream‑to‑Clip
Stream output becomes short clip.

Product requirements:
- `--record` defaults to quick mp4.
- Optional `--gif` export for 5–10s loops.

## 3) Distribution Channels

- **Developer**: GitHub, Hacker News, r/commandline
- **Creative**: r/ASCIIArt, r/terminal, r/Art
- **Video**: short clips on X/Twitter, TikTok, YouTube Shorts

## 4) Messaging Templates

### Short Post (Social)
"Your terminal just learned cinematography. One command → cinematic ASCII.  
`glyph-forge stream https://…`"

### Technical Post
"Glyph Forge turns video into ANSI/ASCII art in real‑time.  
Braille sub‑pixels = 8x resolution per char. 60fps target."

### Community Prompt
"Show your best terminal render. We’ll feature top 3."

## 5) Metrics to Track

- Install → First run conversion rate
- % of runs that export or share
- Time‑to‑wow (seconds to first good output)
- Clip share rate

## 6) Product‑level TODOs (Growth‑Driven)

- [ ] `--demo` flag with bundled asset.
- [ ] `--share` export pipeline (SVG/HTML/PNG).
- [ ] `--copy` for banners (clipboard).
- [ ] Gallery page or `docs/gallery.md`.
- [ ] `--preset` pack names (cinematic, noir, neon, winter).

