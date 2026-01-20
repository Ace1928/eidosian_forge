# Word Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Vocabulary of Eidos.**

## 📖 Overview

`word_forge` processes lexical data, builds semantic networks, and enriches vocabulary.
It integrates with `chromadb` for semantic search and `networkx` for word graphs.

## 🏗️ Architecture
- `src/word_forge/`: Core NLP pipeline.

## 🚀 Usage

```python
from word_forge import Lexicon

lex = Lexicon()
lex.add_word("Eidos", definition="Form, Idea, Essence")
```