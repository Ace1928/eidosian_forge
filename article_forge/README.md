# 💎 Article Forge ⚡

```ascii
    ___    ____  __________  ________    ______   __________  ____  ____________
   /   |  / __ \/_  __/  _/ / ____/ /   / ____/  / ____/ __ \/ __ \/ ____/ ____/
  / /| | / /_/ /  / /  / /  / /   / /   / __/    / /_  / / / / /_/ / / __/ __/   
 / ___ |/ _, _/  / / _/ /  / /___/ /___/ /___   / __/ / /_/ / _, _/ /_/ / /___   
/_/  |_/_/ |_|  /_/ /___/  \____/_____/_____/  /_/    \____/_/ |_|\____/_____/   
                                                                                 
```

> _"The Voice of Eidos. Structuring thought into transmittable intelligence."_

## 🧠 Overview

`article_forge` is the central repository and automation pipeline for written content, blog posts, and articles generated or managed by Eidos. It transforms raw markdown drafts into polished, publishable artifacts (HTML, PDF) using Eidosian workflows. It provides the mechanism for Eidos to project its structured reasoning and systemic insights into the human-readable domain.

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Content Management & Publishing Automation
- **Test Coverage**: ~100% Core Publishing Logic verified.
- **Inference Standard**: **Qwen 3.5 2B** (Used for autonomous editing).
- **Core Architecture**:
  - **`publish.py`**: High-fidelity HTML/PDF rendering engine with Eidosian CSS injection.
  - **`cli/`**: Minimal interface for batch publishing operations.
  - **`scripts/`**: Specialized utilities for biographical profile extraction and VSCode configuration parsing.

## 🚀 Usage & Workflows

### Standard Publishing

Convert a draft into a publishable HTML5 artifact with standard Eidosian typography:
```bash
python -m article_forge.cli publish path/to/draft.md --html-out path/to/draft.html
```

Generate a high-fidelity PDF (requires `weasyprint`):
```bash
python -m article_forge.cli publish path/to/draft.md --pdf-out path/to/report.pdf
```

### Cognitive Profile Extraction

Extract structured JSON identity data from unstructured markdown:
```bash
python scripts/extract_profile_data.py <path_to_biography.md>
```

## 🔗 System Integration

- **Knowledge Forge**: Articles are parsed into concepts and auto-linked within the knowledge graph.
- **LLM Forge**: Supplies the editing substrate for stylistic refinement and "Velvet Beef" alignment.
- **Scribe (Doc Forge)**: Articles are indexed as secondary source material for system documentation.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy documentation into unified standard.
- [x] Integrate Eidosian observability decorators in the publishing pipeline.
- [ ] Implement automated "Content Validation" against the 10 Commandment style guide.

### Future Vector (Phase 3+)
- Build an "Autonomous Journalist" agent that monitors the `event_bus` and generates periodic status articles.
- Add support for interactive HTML artifacts using HTMX and Tailwind integration.
- Implement semantic versioning for individual articles to track narrative evolution.

---
*Generated and maintained by Eidos.*
