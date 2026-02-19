# GraphRAG Qualitative Assessment

- Contract: `graphrag.qualitative.assessment.v1`
- Generated: `2026-02-19T04:44:40.592351+00:00`
- Workspace: `data/graphrag_test/workspace`
- Final score: **0.6514** (rank `B`)
- Deterministic objective: `0.8260`
- Judge score: `0.4991`
- Disagreement: `0.4991`

## Deterministic Scores
- `pipeline_integrity`: `1.0000`
- `workflow_completeness`: `1.0000`
- `entity_coverage`: `0.8571`
- `relationship_density`: `1.0000`
- `community_report_quality`: `0.3500`
- `query_answer_quality`: `0.9500`
- `runtime_score`: `0.5498`

## Stage Scores
- `load_input_documents`: completed=`True` runtime=`0.12142801284790039` score=`0.9973`
- `create_base_text_units`: completed=`True` runtime=`1.866624355316162` score=`0.9585`
- `create_final_documents`: completed=`True` runtime=`0.06017160415649414` score=`0.9987`
- `extract_graph_nlp`: completed=`True` runtime=`12.622421979904175` score=`0.7195`
- `prune_graph`: completed=`True` runtime=`0.05300450325012207` score=`0.9988`
- `finalize_graph`: completed=`True` runtime=`0.07975602149963379` score=`0.9982`
- `create_communities`: completed=`True` runtime=`0.16051530838012695` score=`0.9964`
- `create_final_text_units`: completed=`True` runtime=`0.08317184448242188` score=`0.9982`
- `create_community_reports_text`: completed=`True` runtime=`1.9441554546356201` score=`0.9568`

## Judge Assessments
- `qwen`: factuality=0.9982 grounding=0.9982 coherence=0.9982 usefulness=0.9982 risk_awareness=0.9982
- `llama`: factuality=0.0 grounding=0.0 coherence=0.0 usefulness=0.0 risk_awareness=0.0
