# 💎 Article Forge ⚡

> _"The Voice of Eidos. Structuring thought into transmittable intelligence."_

## 🧠 Overview

`article_forge` is the central repository and automation pipeline for written content, blog posts, and articles generated or managed by Eidos. It transforms raw markdown drafts into polished, publishable artifacts (HTML, PDF) using Eidosian workflows.

```ascii
      ╭───────────────────────────────────────────╮
      │             ARTICLE FORGE                 │
      │    < Markdown -> LLM Edit -> Publish >    │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   SOURCE DRAFTS     │   │   ARTIFACTS     │
      │ (Markdown)          │   │ (HTML / PDF)    │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Content Management & Publishing Automation
- **Test Coverage**: ~100% Core Publishing Logic
- **Architecture**:
  - `src/article_forge/publish.py`: Core logic for HTML/PDF rendering.
  - `src/article_forge/cli/`: Minimal interface for publishing.
  - `scripts/`: Utilities for profile extraction and setup parsing.

## 🚀 Usage & Workflows

### Converting Markdown to HTML/PDF
Convert a draft into a publishable format:
```bash
python -m article_forge.cli publish path/to/draft.md --html-out path/to/draft.html
```

With PDF export (requires `weasyprint` in `eidosian_venv`):
```bash
python -m article_forge.cli publish path/to/draft.md --html-out path/to/draft.html --pdf-out path/to/draft.pdf
```

### Profile Extraction
Extract explicit JSON data from markdown biographies via LLM Forge:
```bash
python scripts/extract_profile_data.py <path_to_about_me.md>
```

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Verify CLI and Python packaging.
- [ ] Ensure 100% type safety (`mypy`).

### Future Vector (Phase 3+)
- Integrate natively with `knowledge_forge` to auto-link concepts in articles.
- Add version history awareness and diff generation for iterative drafts.
- Expand semantic search over generated HTML content.

---
*Generated and maintained by Eidos.*
