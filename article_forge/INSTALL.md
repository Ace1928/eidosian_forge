# 📰 Article Forge Installation

Article generation, summarization, and content management.

## Quick Install

```bash
pip install -e ./article_forge
python -c "from article_forge import ArticleGenerator; print('✓ Ready')"
```

## CLI Usage

```bash
article-forge generate "Topic" --style news
article-forge summarize input.txt --length short
article-forge --help
```

## Dependencies

- `eidosian_core` - Decorators and logging
- LLM integration for generation

