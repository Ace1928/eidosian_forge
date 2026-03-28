from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from eidosian_core import eidosian
from word_forge.parser.language_model import ModelState
from word_forge.parser.structured_validator import ProsodySchema, validated_query

LOGGER = logging.getLogger("word_forge.linguistics.prosody")

@eidosian()
class ProsodyEngine:
    """Generates prosody priors (pitch, duration, intensity) for phrases."""

    def __init__(self, model_state: Optional[ModelState] = None) -> None:
        self.llm_state = model_state

    def generate_priors(
        self, 
        text: str,
        valence: float = 0.0,
        arousal: float = 0.0,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """Generate prosody priors based on text and emotional state."""
        
        # Base rule-based priors
        priors = self._calculate_rule_based_priors(valence, arousal)
        
        # Enrich with LLM if available for "Eidosian" quality
        if self.llm_state:
            llm_priors = self._get_llm_prosody_priors(text, valence, arousal)
            if llm_priors:
                # Blend or overwrite
                priors.update(llm_priors)
                
        return priors

    def _calculate_rule_based_priors(self, valence: float, arousal: float) -> Dict[str, Any]:
        """Heuristic-based prosody calculation."""
        # Pitch: Higher arousal -> Higher pitch
        pitch_multiplier = 1.0 + (arousal * 0.5)
        if valence < -0.5: # Sadness/Depression -> Lower pitch
            pitch_multiplier *= 0.8
            
        # Duration: High arousal -> Faster (lower duration), Low arousal/Negative -> Slower
        duration_multiplier = 1.0 - (arousal * 0.3)
        if valence < -0.3:
            duration_multiplier *= 1.2 # Slower for negative valence
            
        # Intensity: High arousal -> Higher intensity
        intensity_multiplier = 1.0 + (arousal * 0.7)
        
        return {
            "pitch_multiplier": round(pitch_multiplier, 3),
            "duration_multiplier": round(duration_multiplier, 3),
            "intensity_multiplier": round(intensity_multiplier, 3),
            "pause_priors": self._estimate_pauses(arousal, valence)
        }

    def _estimate_pauses(self, arousal: float, valence: float) -> Dict[str, float]:
        """Estimate pause durations between linguistic units."""
        return {
            "comma_pause_ms": 150 * (1.2 if valence < 0 else 1.0),
            "period_pause_ms": 400 * (1.3 if valence < 0 else 1.0),
            "inter_word_ms": 20 * (0.8 if arousal > 0.5 else 1.0)
        }

    def _get_llm_prosody_priors(self, text: str, valence: float, arousal: float) -> Optional[Dict[str, Any]]:
        """Consult the LLM for nuanced prosody parameters."""
        prompt = (
            f"Task: analyze prosody for the phrase '{text}'.\n"
            f"Emotional state: valence={valence}, arousal={arousal}.\n"
            "Return only valid JSON using exactly this schema: "
            '{"pitch_multiplier":float,"duration_multiplier":float,'
            '"intensity_multiplier":float,"pitch_contour":"rising|falling|flat|wavy",'
            '"emphasis_indices":[int]}'
        )
        
        result = validated_query(
            model_state=self.llm_state,
            prompt=prompt,
            context_word=text,
            max_retries=1,
            schema=ProsodySchema,
        )
        
        return result.unwrap() if result.is_success else None
