# Glyph Forge — Demo Recipes

## Rickroll (Local + YouTube)

### One‑shot script

```bash
./scripts/demo_rickroll.sh
```

### Manual (Local file)

```bash
glyph-forge stream assets/rickroll.mp4 --resolution 720p --fps 30 --mode braille --color ansi256 --audio --stats
```

### Manual (YouTube URL)

```bash
glyph-forge stream https://www.youtube.com/watch?v=dQw4w9WgXcQ --resolution 720p --fps 30 --mode braille --color ansi256 --audio --stats
```

### Render-Then-Play (Muxed Audio)

```bash
glyph-forge stream assets/rickroll.mp4 --render-play --resolution 720p --fps 30 --mode braille --color ansi256
```

### Share Exports (GIF/PNG/MP4)

```bash
glyph-forge stream assets/rickroll.mp4 --share gif
glyph-forge stream assets/rickroll.mp4 --share png
glyph-forge stream assets/rickroll.mp4 --share mp4
glyph-forge stream assets/rickroll.mp4 --share apng
glyph-forge stream assets/rickroll.mp4 --share webm
glyph-forge stream assets/rickroll.mp4 --share svg
glyph-forge stream assets/rickroll.mp4 --share html
```

### Playlist Stitching

```bash
glyph-forge stream "https://www.youtube.com/watch?v=ZFj2zhfA4aA&list=PL3zHyzZFGya_nhTUW3cCsWBu56MFY0O4f"
```

## Notes

- Use `glyph-forge doctor` to verify optional dependencies.
- For higher quality, increase resolution/fps; for speed, drop to 480p.
