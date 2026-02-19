# GraphRAG Qualitative Assessment

- Contract: `graphrag.qualitative.assessment.v1`
- Generated: `2026-02-19T04:41:12.821543+00:00`
- Workspace: `data/graphrag_test/workspace`
- Final score: **0.8656** (rank `A`)
- Deterministic objective: `0.8170`
- Judge score: `0.9659`
- Disagreement: `0.0159`

## Deterministic Scores
- `pipeline_integrity`: `1.0000`
- `workflow_completeness`: `1.0000`
- `entity_coverage`: `0.8571`
- `relationship_density`: `1.0000`
- `community_report_quality`: `0.3500`
- `query_answer_quality`: `0.9500`
- `runtime_score`: `0.4589`

## Stage Scores
- `load_input_documents`: completed=`True` runtime=`0.1479816436767578` score=`0.9967`
- `create_base_text_units`: completed=`True` runtime=`1.6325647830963135` score=`0.9637`
- `create_final_documents`: completed=`True` runtime=`0.07093024253845215` score=`0.9984`
- `extract_graph_nlp`: completed=`True` runtime=`15.386770248413086` score=`0.6581`
- `prune_graph`: completed=`True` runtime=`0.07332491874694824` score=`0.9984`
- `finalize_graph`: completed=`True` runtime=`0.1233062744140625` score=`0.9973`
- `create_communities`: completed=`True` runtime=`0.21312642097473145` score=`0.9953`
- `create_final_text_units`: completed=`True` runtime=`0.14603710174560547` score=`0.9968`
- `create_community_reports_text`: completed=`True` runtime=`2.101411819458008` score=`0.9533`

## Judge Assessments
- `qwen`: factuality=0.9967 grounding=0.9637 coherence=0.9984 usefulness=0.9973 risk_awareness=0.9533
- `llama`: factuality=0.95 grounding=0.95 coherence=0.95 usefulness=0.95 risk_awareness=0.95
