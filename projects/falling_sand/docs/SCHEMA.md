# Index Schema

The index document is versioned to support seamless migrations as metadata grows.

## Current version
- `schema_version`: 3

## Migration
Use `falling_sand.schema.migrate_document_dict` to upgrade older JSON payloads to the latest schema version.

## Compatibility
New fields are appended with optional values (`null`) so older consumers can ignore additions.

## Benchmark schema
`benchmark_summary` now contains a `benchmarks` list, each entry with a `name` and timing stats.
