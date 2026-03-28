from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Protocol, Tuple

from eidosian_core import eidosian
from word_forge.parser.language_model import ModelState
from word_forge.parser.structured_validator import validated_query

LOGGER = logging.getLogger("word_forge.linguistics.g2p")

class G2PResult(Dict[str, Optional[str]]):
    """Phonetic transcription result."""
    ipa: str
    arpabet: Optional[str]
    stress_pattern: Optional[str]

class G2PProvider(Protocol):
    """Protocol for G2P conversion engines."""
    def convert(self, text: str, context: Optional[str] = None) -> G2PResult: ...

@eidosian()
class LLMG2PProvider:
    """LLM-based G2P conversion with schema validation."""

    def __init__(self, model_state: ModelState) -> None:
        self.model_state = model_state

    def convert(self, text: str, context: Optional[str] = None) -> G2PResult:
        """Use LLM to generate high-fidelity phonetic transcriptions."""
        prompt = (
            f"Transcribe the following term into IPA and Arpabet phonetics.\n"
            f"Term: '{text}'\n"
            f"Context: {context if context else 'General terminology'}\n\n"
            "Provide a JSON object with the following schema:\n"
            "{\n"
            '  "ipa": "string (International Phonetic Alphabet)",\n'
            '  "arpabet": "string (Arpabet representation)",\n'
            '  "stress_pattern": "string (e.g., 1-0 for primary-none)"\n'
            "}\n"
            "Return ONLY valid JSON."
        )

        result = validated_query(
            model_state=self.model_state,
            prompt=prompt,
            context_word=text,
            max_retries=2
        )

        if result.is_success:
            data = result.unwrap()
            return G2PResult(
                ipa=data.get("ipa", ""),
                arpabet=data.get("arpabet"),
                stress_pattern=data.get("stress_pattern")
            )
        
        # Fallback to empty results if LLM fails
        return G2PResult(ipa="", arpabet=None, stress_pattern=None)

@eidosian()
class G2PManager:
    """Orchestrates G2P conversion across different providers and handles exceptions."""

    def __init__(
        self,
        primary_provider: Optional[G2PProvider] = None,
        exceptions: Optional[Dict[str, G2PResult]] = None
    ) -> None:
        self.primary = primary_provider
        self.exceptions = exceptions or {}
        # Pre-load some common exceptions or domain terms if needed
        self._load_default_exceptions()

    def _load_default_exceptions(self) -> None:
        """Load foundational G2P exceptions."""
        # Example exceptions
        self.exceptions["eidos"] = G2PResult(
            ipa="ˈaɪ.dɒs",
            arpabet="AY1 D AA0 S",
            stress_pattern="1-0"
        )

    def convert(self, text: str, context: Optional[str] = None) -> G2PResult:
        """Convert text to phonetics, checking exceptions first."""
        normalized_text = text.lower().strip()
        
        if normalized_text in self.exceptions:
            return self.exceptions[normalized_text]
        
        if self.primary:
            return self.primary.convert(normalized_text, context)
            
        return G2PResult(ipa="", arpabet=None, stress_pattern=None)

    def add_exception(self, text: str, result: G2PResult) -> None:
        """Manually add a G2P exception."""
        self.exceptions[text.lower().strip()] = result
