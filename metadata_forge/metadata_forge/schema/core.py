"""
Core schema components for the Eidosian metadata system.

This module implements the fundamental structures that define
and validate metadata across all Eidosian repositories.
"""
from typing import Dict, List, Any, Optional, Union, TypedDict, cast
from dataclasses import dataclass, field
import json
from datetime import datetime

# Type definitions
class ParameterDef(TypedDict):
    name: str
    type: str
    optional: bool
    default: Optional[Any]
    description: str

class ReturnDef(TypedDict):
    type: str
    description: str

class TestVerificationDef(TypedDict, total=False):
    unit_tests: Optional[str]
    integration_tests: Optional[str]
    edge_case_coverage: Optional[List[str]]

class PerformanceProfileDef(TypedDict, total=False):
    execution_time: Optional[str]
    memory_usage: Optional[str]
    cpu_usage: Optional[str]
    recursion_complexity: Optional[str]

class BehavioralNotesDef(TypedDict, total=False):
    concurrency_handling: Optional[str]
    error_handling: Optional[str]
    additional_notes: Optional[str]

class InteractionsDef(TypedDict, total=False):
    calls: List[str]
    called_by: List[str]
    depends_on: List[str]
    modifies: List[str]

class ModifiabilityDef(TypedDict, total=False):
    expansion_notes: Optional[str]
    composable_with: Optional[List[str]]

class TraceabilityDef(TypedDict, total=False):
    change_log: Optional[str]
    commit_hash: Optional[str]
    tag: Optional[str]
    time_since_last_edit: Optional[str]

class EntityMetadata(TypedDict):
    entity: str  # Required: e.g., "Function", "Class", "Module", etc.
    identifier: str  # Required: Unique identifier for the artifact.
    version: str  # Required: Semantic version, commit hash, or timestamp.
    purpose: str  # Required: What the artifact does
    context: str  # Required: How it fits into the larger system
    parameters: List[ParameterDef]  # Required for functions/scripts
    returns: ReturnDef  # Required: Describes the return value.
    test_verification: Optional[TestVerificationDef]
    performance_profile: Optional[PerformanceProfileDef]
    behavioral_notes: Optional[BehavioralNotesDef]
    interactions: Optional[InteractionsDef]
    modifiability: Optional[ModifiabilityDef]
    programmatic_traceability: Optional[TraceabilityDef]

@dataclass
class MetadataField:
    """Definition of a metadata field with validation rules."""
    field_name: str
    data_type: str
    requirement: str  # "Required" or "Optional"
    description: str
    allowed_values: Optional[List[str]] = None
    example: Optional[Any] = None

@dataclass
class SchemaDefinition:
    """Complete metadata schema definition."""
    schema_entity: str = "Metadata Schema"
    schema_identifier: str = field(default_factory=lambda: f"eidosian_metadata_schema_v1.0_{int(datetime.now().timestamp())}")
    schema_version: str = "v1.0.0"
    schema_purpose: str = "Provides a universal, self-documenting metadata framework for code artifacts"
    schema_context: str = "Operates within system-wide introspection and automated analysis framework"
    schema_fields: List[MetadataField] = field(default_factory=list)
    schema_modifiability: Dict[str, Any] = field(default_factory=dict)
    schema_traceability: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema definition to dictionary format."""
        return {
            "schema_entity": self.schema_entity,
            "schema_identifier": self.schema_identifier,
            "schema_version": self.schema_version,
            "schema_purpose": self.schema_purpose,
            "schema_context": self.schema_context,
            "schema_fields": [
                {
                    "field_name": field.field_name,
                    "data_type": field.data_type,
                    "requirement": field.requirement,
                    "description": field.description,
                    "allowed_values": field.allowed_values,
                    "example": field.example
                }
                for field in self.schema_fields
            ],
            "schema_modifiability": self.schema_modifiability,
            "schema_traceability": self.schema_traceability
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert schema definition to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

def create_metadata_template() -> EntityMetadata:
    """Create an empty metadata template with all required fields."""
    return cast(EntityMetadata, {
        "entity": "",
        "identifier": "",
        "version": "",
        "purpose": "",
        "context": "",
        "parameters": [],
        "returns": {
            "type": "",
            "description": ""
        },
        "test_verification": {
            "unit_tests": None,
            "integration_tests": None,
            "edge_case_coverage": None
        },
        "performance_profile": {
            "execution_time": None,
            "memory_usage": None,
            "cpu_usage": None,
            "recursion_complexity": None
        },
        "behavioral_notes": {
            "concurrency_handling": None,
            "error_handling": None,
            "additional_notes": None
        },
        "interactions": {
            "calls": [],
            "called_by": [],
            "depends_on": [],
            "modifies": []
        },
        "modifiability": {
            "expansion_notes": None,
            "composable_with": None
        },
        "programmatic_traceability": {
            "change_log": None,
            "commit_hash": None,
            "tag": None,
            "time_since_last_edit": None
        }
    })

def validate_metadata(metadata: EntityMetadata) -> Dict[str, List[str]]:
    """
    Validate metadata against schema requirements.
    
    Args:
        metadata: The metadata dictionary to validate
        
    Returns:
        Dictionary of errors by field, empty if valid
    """
    errors: Dict[str, List[str]] = {}
    
    # Required fields
    required_fields = ["entity", "identifier", "version", "purpose", "context", "returns"]
    for field in required_fields:
        if field not in metadata or not metadata[field]:
            if field not in errors:
                errors[field] = []
            errors[field].append(f"{field} is required")
    
    # Entity type
    if "entity" in metadata and metadata["entity"]:
        allowed_entities = [
            "Function", "Class", "Module", "Script", "Configuration", 
            "Test", "API Endpoint", "Constant", "Documentation", "Example Code"
        ]
        if metadata["entity"] not in allowed_entities:
            if "entity" not in errors:
                errors["entity"] = []
            errors["entity"].append(f"Entity must be one of: {', '.join(allowed_entities)}")
    
    # Returns field structure
    if "returns" in metadata and metadata["returns"]:
        returns = metadata["returns"]
        if not isinstance(returns, dict):
            if "returns" not in errors:
                errors["returns"] = []
            errors["returns"].append("Returns must be a dictionary")
        else:
            for required in ["type", "description"]:
                if required not in returns or not returns[required]:
                    if "returns" not in errors:
                        errors["returns"] = []
                    errors["returns"].append(f"Returns.{required} is required")
    
    # Parameters validation for functions
    if metadata.get("entity") in ["Function", "Script"]:
        if "parameters" not in metadata or not isinstance(metadata["parameters"], list):
            if "parameters" not in errors:
                errors["parameters"] = []
            errors["parameters"].append("Parameters must be a list for functions and scripts")
        else:
            for i, param in enumerate(metadata["parameters"]):
                if not isinstance(param, dict):
                    if "parameters" not in errors:
                        errors["parameters"] = []
                    errors["parameters"].append(f"Parameter {i} must be a dictionary")
                    continue
                
                for required in ["name", "type", "description"]:
                    if required not in param or not param[required]:
                        if "parameters" not in errors:
                            errors["parameters"] = []
                        errors["parameters"].append(f"Parameter {i}.{required} is required")
                
                if "optional" in param and not isinstance(param["optional"], bool):
                    if "parameters" not in errors:
                        errors["parameters"] = []
                    errors["parameters"].append(f"Parameter {i}.optional must be a boolean")
    
    return errors
