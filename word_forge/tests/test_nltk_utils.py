"""Tests for NLTK data management using real downloads."""

from __future__ import annotations

from word_forge.utils.nltk_utils import ensure_nltk_data


def test_ensure_nltk_data_runs() -> None:
    downloaded = ensure_nltk_data()
    assert isinstance(downloaded, list)
