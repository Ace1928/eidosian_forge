#!/usr/bin/env python3
"""
OCR Engine with Region Caching

Performs OCR only on changed regions for efficiency.
Maintains a spatial index of detected text.

Author: Eidos
Version: 1.0.0
"""

import subprocess
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
import hashlib

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    Image = None

from .screen_state import BoundingBox, ScreenFrame, ChangeRegion


@dataclass
class TextRegion:
    """A region of detected text."""
    text: str
    bbox: BoundingBox
    confidence: float
    timestamp: float = field(default_factory=time.time)
    frame_hash: str = ""
    
    def matches(self, query: str, case_sensitive: bool = False) -> bool:
        """Check if text matches query."""
        if case_sensitive:
            return query in self.text
        return query.lower() in self.text.lower()


@dataclass 
class OCRResult:
    """Result of OCR operation."""
    success: bool
    text_regions: List[TextRegion]
    full_text: str
    duration_ms: float
    region_processed: Optional[BoundingBox] = None


class OCREngine:
    """Efficient OCR engine with region caching."""
    
    def __init__(self, cache_size: int = 50):
        self.available = TESSERACT_AVAILABLE
        self._cache: Dict[str, List[TextRegion]] = {}
        self._cache_order: List[str] = []
        self._cache_size = cache_size
        self._tesseract_config = '--oem 3 --psm 6'
        self.stats = {"ocr_calls": 0, "cache_hits": 0, "total_text_regions": 0}
    
    def _get_region_hash(self, image, bbox: Optional[BoundingBox]) -> str:
        """Generate hash for image region."""
        if bbox and image:
            region = image.crop(bbox.to_tuple())
        else:
            region = image
        
        if not region:
            return ""
        
        small = region.resize((64, 64)).convert('L')
        data = small.tobytes()
        return hashlib.md5(data).hexdigest()
    
    def _add_to_cache(self, key: str, regions: List[TextRegion]):
        """Add to cache with LRU eviction."""
        if key in self._cache:
            self._cache_order.remove(key)
        elif len(self._cache) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = regions
        self._cache_order.append(key)
    
    def ocr_frame(self, frame: ScreenFrame, 
                  region: Optional[BoundingBox] = None,
                  use_cache: bool = True) -> OCRResult:
        """Perform OCR on frame or region."""
        if not self.available or not frame.image:
            return OCRResult(False, [], "", 0, region)
        
        start_time = time.time()
        
        if region:
            try:
                img = frame.image.crop(region.to_tuple())
            except:
                img = frame.image
        else:
            img = frame.image
        
        region_hash = self._get_region_hash(img, None)
        
        if use_cache and region_hash in self._cache:
            self.stats["cache_hits"] += 1
            cached = self._cache[region_hash]
            full_text = "\n".join(r.text for r in cached)
            return OCRResult(True, cached, full_text, (time.time() - start_time) * 1000, region)
        
        self.stats["ocr_calls"] += 1
        
        try:
            data = pytesseract.image_to_data(
                img, config=self._tesseract_config, output_type=pytesseract.Output.DICT
            )
            
            text_regions = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf > 30:
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    if region:
                        x += region.x
                        y += region.y
                    
                    text_regions.append(TextRegion(
                        text=text,
                        bbox=BoundingBox(x, y, w, h),
                        confidence=conf / 100.0,
                        frame_hash=frame.hash
                    ))
            
            self._add_to_cache(region_hash, text_regions)
            self.stats["total_text_regions"] += len(text_regions)
            full_text = " ".join(r.text for r in text_regions)
            
            return OCRResult(True, text_regions, full_text, (time.time() - start_time) * 1000, region)
            
        except Exception as e:
            return OCRResult(False, [], str(e), (time.time() - start_time) * 1000, region)
    
    def ocr_changed_regions(self, frame: ScreenFrame, changes: List[ChangeRegion]) -> List[OCRResult]:
        """OCR only the changed regions."""
        results = []
        for change in changes:
            expanded = BoundingBox(
                x=max(0, change.bbox.x - 10),
                y=max(0, change.bbox.y - 10),
                width=change.bbox.width + 20,
                height=change.bbox.height + 20
            )
            results.append(self.ocr_frame(frame, expanded))
        return results
    
    def find_text(self, frame: ScreenFrame, query: str, case_sensitive: bool = False) -> List[TextRegion]:
        """Find all occurrences of text on screen."""
        result = self.ocr_frame(frame)
        if not result.success:
            return []
        return [r for r in result.text_regions if r.matches(query, case_sensitive)]
    
    def find_text_location(self, frame: ScreenFrame, query: str) -> Optional[Tuple[int, int]]:
        """Find center position of text on screen."""
        matches = self.find_text(frame, query)
        return matches[0].bbox.center if matches else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get OCR engine statistics."""
        total = self.stats["ocr_calls"] + self.stats["cache_hits"]
        return {
            **self.stats,
            "cache_size": len(self._cache),
            "cache_hit_rate": self.stats["cache_hits"] / max(1, total)
        }
    
    def clear_cache(self):
        """Clear OCR cache."""
        self._cache.clear()
        self._cache_order.clear()
