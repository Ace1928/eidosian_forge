"""
👁️ Visual Feedback Module

Provides screen state analysis and change detection for automation verification.
Built on PIL for image processing without external API dependencies.

Created: 2026-01-23
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageChops, ImageStat
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from .wayland_control import take_screenshot


@dataclass
class ScreenState:
    """Represents a captured screen state."""
    path: Path
    timestamp: str
    width: int
    height: int
    hash: str  # perceptual hash for quick comparison
    
    @classmethod
    def capture(cls, output_dir: Optional[Path] = None) -> Optional["ScreenState"]:
        """Capture current screen state."""
        if output_dir is None:
            output_dir = Path("/tmp/eidos_screenshots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_dir / f"screen_{timestamp}.png"
        
        result = take_screenshot(str(output_path))
        if not result.get("success"):
            return None
        
        # Get image info
        if _PIL_AVAILABLE:
            img = Image.open(output_path)
            width, height = img.size
            phash = _perceptual_hash(img)
            img.close()
        else:
            width, height = 0, 0
            phash = _file_hash(output_path)
        
        return cls(
            path=output_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
            width=width,
            height=height,
            hash=phash
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "hash": self.hash
        }


def _file_hash(path: Path) -> str:
    """Simple file hash."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _perceptual_hash(img: "Image.Image", size: int = 8) -> str:
    """
    Compute perceptual hash (pHash) for an image.
    Similar images produce similar hashes.
    """
    # Resize to small size and convert to grayscale
    img = img.resize((size, size), Image.Resampling.LANCZOS).convert("L")
    
    # Get pixel data
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    
    # Create hash based on whether each pixel is above/below average
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return format(int(bits, 2), "016x")


def compare_states(state1: ScreenState, state2: ScreenState) -> Dict[str, Any]:
    """
    Compare two screen states and return detailed analysis.
    """
    result = {
        "states": [state1.to_dict(), state2.to_dict()],
        "hash_match": state1.hash == state2.hash,
        "timestamp_delta_ms": _time_delta_ms(state1.timestamp, state2.timestamp)
    }
    
    if not _PIL_AVAILABLE:
        result["pixel_analysis"] = None
        result["error"] = "PIL not available for detailed analysis"
        return result
    
    # Load images
    img1 = Image.open(state1.path)
    img2 = Image.open(state2.path)
    
    # Size check
    if img1.size != img2.size:
        result["size_match"] = False
        result["pixel_analysis"] = None
        return result
    
    result["size_match"] = True
    
    # Compute difference
    diff = ImageChops.difference(img1, img2)
    stat = ImageStat.Stat(diff)
    
    # Calculate metrics
    max_diff = max(stat.extrema, key=lambda x: x[1])[1]  # max across channels
    mean_diff = sum(stat.mean) / len(stat.mean)
    
    # Count changed pixels (threshold-based)
    threshold = 10
    changed_mask = diff.convert("L").point(lambda x: 255 if x > threshold else 0)
    changed_pixels = sum(1 for p in changed_mask.getdata() if p > 0)
    total_pixels = img1.width * img1.height
    change_percent = (changed_pixels / total_pixels) * 100
    
    result["pixel_analysis"] = {
        "max_difference": max_diff,
        "mean_difference": round(mean_diff, 2),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_percent": round(change_percent, 4),
        "has_significant_change": change_percent > 0.5  # >0.5% pixels changed
    }
    
    img1.close()
    img2.close()
    
    return result


def _time_delta_ms(ts1: str, ts2: str) -> float:
    """Calculate time delta between ISO timestamps in milliseconds."""
    from datetime import datetime
    t1 = datetime.fromisoformat(ts1.replace("Z", "+00:00"))
    t2 = datetime.fromisoformat(ts2.replace("Z", "+00:00"))
    return abs((t2 - t1).total_seconds() * 1000)


def wait_for_change(
    initial_state: ScreenState,
    timeout_sec: float = 10.0,
    check_interval_sec: float = 0.5,
    change_threshold_percent: float = 0.5
) -> Optional[ScreenState]:
    """
    Wait for screen to change from initial state.
    
    Returns new state if change detected, None if timeout.
    """
    start = time.time()
    
    while (time.time() - start) < timeout_sec:
        current = ScreenState.capture()
        if current is None:
            time.sleep(check_interval_sec)
            continue
        
        comparison = compare_states(initial_state, current)
        
        if comparison.get("pixel_analysis", {}).get("change_percent", 0) >= change_threshold_percent:
            return current
        
        # Clean up intermediate screenshot if no change
        if current.path.exists() and current.hash == initial_state.hash:
            current.path.unlink()
        
        time.sleep(check_interval_sec)
    
    return None


def get_pixel_color(x: int, y: int, screenshot_path: Optional[str] = None) -> Optional[Tuple[int, int, int]]:
    """
    Get RGB color of pixel at (x, y).
    If screenshot_path not provided, takes a new screenshot.
    """
    if not _PIL_AVAILABLE:
        return None
    
    if screenshot_path is None:
        state = ScreenState.capture()
        if state is None:
            return None
        screenshot_path = str(state.path)
    
    img = Image.open(screenshot_path)
    
    if 0 <= x < img.width and 0 <= y < img.height:
        pixel = img.getpixel((x, y))
        img.close()
        return pixel[:3] if len(pixel) > 3 else pixel
    
    img.close()
    return None


def find_color_region(
    color: Tuple[int, int, int],
    tolerance: int = 20,
    screenshot_path: Optional[str] = None
) -> List[Tuple[int, int]]:
    """
    Find all pixels matching a color within tolerance.
    Returns list of (x, y) coordinates.
    
    Warning: Can be slow for full screenshots.
    """
    if not _PIL_AVAILABLE:
        return []
    
    if screenshot_path is None:
        state = ScreenState.capture()
        if state is None:
            return []
        screenshot_path = str(state.path)
    
    img = Image.open(screenshot_path)
    pixels = img.load()
    
    matches = []
    for y in range(0, img.height, 4):  # Sample every 4th pixel for speed
        for x in range(0, img.width, 4):
            p = pixels[x, y]
            if all(abs(p[i] - color[i]) <= tolerance for i in range(3)):
                matches.append((x, y))
    
    img.close()
    return matches


# Export
__all__ = [
    "ScreenState",
    "compare_states",
    "wait_for_change",
    "get_pixel_color",
    "find_color_region"
]


# OCR Integration using Tesseract
import subprocess
import tempfile


def ocr_screenshot(screenshot_path: Optional[str] = None, lang: str = "eng") -> Dict[str, Any]:
    """
    Extract text from screenshot using Tesseract OCR.
    
    Args:
        screenshot_path: Path to image. If None, captures new screenshot.
        lang: Tesseract language code (default: eng)
    
    Returns:
        Dict with extracted text and confidence
    """
    if screenshot_path is None:
        state = ScreenState.capture()
        if state is None:
            return {"success": False, "error": "Failed to capture screenshot"}
        screenshot_path = str(state.path)
    
    try:
        result = subprocess.run(
            ["tesseract", screenshot_path, "stdout", "-l", lang],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            text = result.stdout.strip()
            return {
                "success": True,
                "text": text,
                "lines": text.split("\n") if text else [],
                "line_count": len(text.split("\n")) if text else 0,
                "char_count": len(text),
                "source": screenshot_path
            }
        else:
            return {"success": False, "error": result.stderr}
            
    except FileNotFoundError:
        return {"success": False, "error": "tesseract not found"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "OCR timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def ocr_region(
    x: int, y: int, width: int, height: int,
    screenshot_path: Optional[str] = None,
    lang: str = "eng"
) -> Dict[str, Any]:
    """
    Extract text from a specific region of the screen.
    
    Args:
        x, y: Top-left corner of region
        width, height: Size of region
        screenshot_path: Optional existing screenshot
        lang: Tesseract language
    
    Returns:
        Dict with extracted text from region
    """
    if not _PIL_AVAILABLE:
        return {"success": False, "error": "PIL not available"}
    
    # Get screenshot
    if screenshot_path is None:
        state = ScreenState.capture()
        if state is None:
            return {"success": False, "error": "Failed to capture screenshot"}
        screenshot_path = str(state.path)
    
    try:
        # Crop region
        img = Image.open(screenshot_path)
        region = img.crop((x, y, x + width, y + height))
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            region.save(f, "PNG")
            temp_path = f.name
        
        img.close()
        region.close()
        
        # OCR the region
        result = ocr_screenshot(temp_path, lang)
        result["region"] = {"x": x, "y": y, "width": width, "height": height}
        
        # Clean up
        Path(temp_path).unlink()
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def find_text_on_screen(
    search_text: str,
    screenshot_path: Optional[str] = None,
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """
    Check if specific text appears on screen.
    
    Note: Does not return position (that requires more complex OCR with bounding boxes).
    
    Args:
        search_text: Text to find
        screenshot_path: Optional existing screenshot
        case_sensitive: Whether to match case
    
    Returns:
        Dict with found status and context
    """
    ocr_result = ocr_screenshot(screenshot_path)
    
    if not ocr_result.get("success"):
        return ocr_result
    
    full_text = ocr_result["text"]
    
    if not case_sensitive:
        found = search_text.lower() in full_text.lower()
    else:
        found = search_text in full_text
    
    return {
        "success": True,
        "found": found,
        "search_text": search_text,
        "total_text_length": len(full_text)
    }


# Update exports
__all__.extend(["ocr_screenshot", "ocr_region", "find_text_on_screen"])
