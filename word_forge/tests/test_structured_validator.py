from __future__ import annotations

from word_forge.parser.structured_validator import (
    G2PSchema,
    PhraseExtractionSchema,
    ProsodySchema,
    StructuredValidator,
)


def test_phrase_extraction_schema_accepts_expected_shape() -> None:
    validator = StructuredValidator(schema=PhraseExtractionSchema)
    result = validator.validate(
        '{"phrases": ["Atlas dashboard", "Word Forge"], "entities": ["Eidos", "Code Forge"]}',
        context_word="atlas",
    )
    assert result.is_success
    payload = result.unwrap()
    assert payload["phrases"] == ["Atlas dashboard", "Word Forge"]
    assert payload["entities"] == ["Eidos", "Code Forge"]


def test_phrase_extraction_schema_repairs_string_lists() -> None:
    validator = StructuredValidator(schema=PhraseExtractionSchema)
    result = validator.validate('{"phrases": "Atlas dashboard; Word Forge", "entities": "Eidos, Code Forge"}')
    assert result.is_success
    payload = result.unwrap()
    assert payload["phrases"] == ["Atlas dashboard", "Word Forge"]
    assert payload["entities"] == ["Eidos", "Code Forge"]


def test_g2p_schema_accepts_g2p_payload() -> None:
    validator = StructuredValidator(schema=G2PSchema)
    result = validator.validate(
        '{"ipa": "/sɛrənˈdɪpəti/", "arpabet": "S EH R AH N D IH P AH T IY", "stress_pattern": "0-0-1-0-0"}',
        context_word="serendipity",
    )
    assert result.is_success
    payload = result.unwrap()
    assert payload["ipa"] == "/sɛrənˈdɪpəti/"


def test_prosody_schema_coerces_emphasis_indices() -> None:
    validator = StructuredValidator(schema=ProsodySchema)
    result = validator.validate(
        '{"pitch_multiplier": "1.1", "duration_multiplier": 0.9, "intensity_multiplier": 1.2, '
        '"pitch_contour": "rising", "emphasis_indices": ["1", 3, "oops"]}'
    )
    assert result.is_success
    payload = result.unwrap()
    assert payload["pitch_multiplier"] == 1.1
    assert payload["emphasis_indices"] == [1, 3]


def test_g2p_schema_rejects_placeholder_echo() -> None:
    validator = StructuredValidator(schema=G2PSchema)
    result = validator.validate(
        '{"ipa": "string (International Phonetic Alphabet)", '
        '"arpabet": "string (Arpabet representation)", '
        '"stress_pattern": "string (e.g., 1-0 for primary-none)"}',
        context_word="serendipity",
    )
    assert result.is_failure
