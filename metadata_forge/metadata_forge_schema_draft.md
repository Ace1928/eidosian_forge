# Universal Eidosian Metadata Template
# This template is used to annotate any code artifact (Function, Class, Module, etc.)
# Every field is explicitly present; optional fields have a default value of None.
# This ensures uniformity for programmatic analysis, linking, and evolution.

UNIVERSAL_EIDOSIAN_METADATA_TEMPLATE = {
    "entity": "Function",  # Required: e.g., "Function", "Class", "Module", etc.
    "identifier": "generate_embeddings",  # Required: Unique identifier for the artifact.
    "version": "v1.2.3 (Commit: abc123)",  # Required: Semantic version, commit hash, or timestamp.
    "purpose": "Transforms input text into a numerical vector representation for ML tasks.",  # Required.
    "context": "Core component in the embedding pipeline; utilized by retrieval and ranking modules.",  # Required.
    "parameters": [  # Required for functions/scripts; otherwise use an empty list.
        {
            "name": "text",              # Required: Parameter name.
            "type": "str",               # Required: Data type.
            "optional": False,           # Required: Boolean flag.
            "default": None,             # Optional: Default value if applicable (None if not).
            "description": "Input string to be transformed into an embedding."  # Required.
        },
        {
            "name": "model",
            "type": "str",
            "optional": True,
            "default": "nomic-embed-text",
            "description": "Identifier for the embedding model to be used."
        },
        {
            "name": "normalize",
            "type": "bool",
            "optional": True,
            "default": True,
            "description": "Flag to indicate if the output vector should be normalized."
        }
    ],
    "returns": {  # Required: Describes the return value.
        "type": "List[float]",  # Required.
        "description": "The generated embedding vector representing the input text."  # Required.
    },
    "test_verification": {  # Optional: Testing coverage and verification details.
        "unit_tests": "100% coverage (see test_generate_embeddings.py)",  # Required if tests exist.
        "integration_tests": None,  # Optional.
        "edge_case_coverage": None  # Optional: List of strings; use None if not applicable.
    },
    "performance_profile": {  # Optional: Performance metrics.
        "execution_time": "12.3ms (median over 1000 runs)",  # Required if applicable.
        "memory_usage": "3.5MB (average)",  # Required if applicable.
        "cpu_usage": None,  # Optional.
        "recursion_complexity": None  # Optional: e.g., "O(n) depth" or None.
    },
    "behavioral_notes": {  # Optional: Runtime, concurrency, and error-handling details.
        "concurrency_handling": "Function is stateless and thread-safe",  # Optional.
        "error_handling": "Raises ValueError on empty input",  # Required if error conditions are managed.
        "additional_notes": None  # Optional.
    },
    "interactions": {  # Optional: Cross-references to related artifacts.
        "calls": ["fetch_model_weights", "normalize_vector"],  # Optional: List of identifiers.
        "called_by": ["semantic_search", "generate_recommendations"],  # Optional.
        "depends_on": ["nomic-embed-text model"],  # Optional.
        "modifies": []  # Optional: List, empty if no modifications.
    },
    "modifiability": {  # Optional: Guidelines for extension/integration.
        "expansion_notes": "Can be extended to support additional models and normalization techniques.",  # Optional.
        "composable_with": ["text_preprocessing", "embedding_postprocessing"]  # Optional.
    },
    "programmatic_traceability": {  # Optional: Change tracking and version control.
        "change_log": "Optimized vector normalization routine.",  # Required if modifications are tracked.
        "commit_hash": "abc123",  # Optional.
        "tag": None,  # Optional.
        "time_since_last_edit": "3 days ago"  # Optional.
    }
}

# ------------------------------------------------------------------------------
# Meta-Metadata Template
# This template defines the metadata schema for the metadata standard itself.
# Every field is explicitly specified; optional fields default to None.

EIDOSIAN_META_METADATA_TEMPLATE = {
    "schema_entity": "Metadata Schema",  # Required: Should be "Metadata Schema".
    "schema_identifier": "Universal_Eidosian_Metadata_Schema_v1.0",  # Required: Unique identifier.
    "schema_version": "v1.0.0",  # Required: Version info (semantic, commit, or timestamp).
    "schema_purpose": "Provides a universal, self-documenting, self-verifying, and self-optimizing framework for all code artifacts.",  # Required.
    "schema_context": "Operates within a system-wide introspection and automated analysis framework, enabling recursive refinement.",  # Required.
    "schema_fields": [  # Required: List of all field definitions in the metadata schema.
        {
            "field_name": "entity",
            "data_type": "str",
            "requirement": "Required",
            "description": "Type of artifact (e.g., 'Function', 'Module').",
            "allowed_values": ["Function", "Class", "Module", "Script", "Configuration", "Test", "API Endpoint", "Constant", "Documentation", "Example Code"],
            "example": "Function"
        },
        {
            "field_name": "identifier",
            "data_type": "str",
            "requirement": "Required",
            "description": "A unique identifier for the artifact.",
            "allowed_values": None,
            "example": "generate_embeddings"
        },
        {
            "field_name": "version",
            "data_type": "str",
            "requirement": "Required",
            "description": "Semantic versioning, commit hash, or timestamp.",
            "allowed_values": None,
            "example": "v1.2.3 (Commit: abc123)"
        },
        {
            "field_name": "purpose",
            "data_type": "str",
            "requirement": "Required",
            "description": "Concise explanation of the artifact’s intent.",
            "allowed_values": None,
            "example": "Transforms input text into an embedding vector."
        },
        {
            "field_name": "context",
            "data_type": "str",
            "requirement": "Required",
            "description": "How the artifact fits into the broader system.",
            "allowed_values": None,
            "example": "Core component in the embedding pipeline."
        },
        {
            "field_name": "parameters",
            "data_type": "List[Dict[str, Any]]",
            "requirement": "Optional",
            "description": "Definitions for input parameters.",
            "allowed_values": None,
            "example": None
        },
        {
            "field_name": "returns",
            "data_type": "Dict[str, str]",
            "requirement": "Required",
            "description": "Description of the return value.",
            "allowed_values": None,
            "example": {"type": "List[float]", "description": "Generated embedding vector."}
        },
        {
            "field_name": "test_verification",
            "data_type": "Dict[str, Any]",
            "requirement": "Optional",
            "description": "Details on test coverage and verification status.",
            "allowed_values": None,
            "example": None
        },
        {
            "field_name": "performance_profile",
            "data_type": "Dict[str, Any]",
            "requirement": "Optional",
            "description": "Metrics related to performance profiling.",
            "allowed_values": None,
            "example": None
        },
        {
            "field_name": "behavioral_notes",
            "data_type": "Dict[str, Any]",
            "requirement": "Optional",
            "description": "Runtime, concurrency, or error-handling details.",
            "allowed_values": None,
            "example": None
        },
        {
            "field_name": "interactions",
            "data_type": "Dict[str, List[str]]",
            "requirement": "Optional",
            "description": "Cross-references to related artifacts.",
            "allowed_values": None,
            "example": None
        },
        {
            "field_name": "modifiability",
            "data_type": "Dict[str, Any]",
            "requirement": "Optional",
            "description": "Guidelines for extending or composing this artifact.",
            "allowed_values": None,
            "example": None
        },
        {
            "field_name": "programmatic_traceability",
            "data_type": "Dict[str, Any]",
            "requirement": "Optional",
            "description": "Data for change tracking and version control.",
            "allowed_values": None,
            "example": None
        }
    ],
    "schema_modifiability": {  # Optional meta-data for this schema.
        "expansion_notes": None,
        "composable_with": None
    },
    "schema_traceability": {  # Optional meta-data for version tracking.
        "change_log": None,
        "commit_hash": None,
        "tag": None,
        "time_since_last_edit": None
    }
}

# Example usage: linking an artifact to its metadata.
def print_metadata_info(metadata):
    for key, value in metadata.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    print("=== Universal Eidosian Metadata Template ===")
    print_metadata_info(UNIVERSAL_EIDOSIAN_METADATA_TEMPLATE)
    print("\n=== Eidosian Meta-Metadata Template ===")
    print_metadata_info(EIDOSIAN_META_METADATA_TEMPLATE)
