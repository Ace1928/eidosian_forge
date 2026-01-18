"""
Content templates for different response sections.

This module provides templates for generating structured content
for different sections of model responses, allowing for consistent
formatting and presentation across various model analyses.
"""

from typing import Dict, Final, Literal

# Section key type for improved type safety
SectionKey = Literal[
    "overview",
    "technical_details",
    "advantages",
    "limitations",
    "model_architecture",
    "training_data",
    "strengths_weaknesses",
    "use_cases",
]

# Templates for each section type, with placeholders for model-specific details
SECTION_TEMPLATES: Final[Dict[str, str]] = {
    "overview": (
        "{model} is a {architecture} developed by {company} around {year}. "
        "It represents one of the significant advancements in the field of natural "
        "language processing for {topic}. The model demonstrates impressive capabilities "
        "in understanding and generating human-like text with context awareness."
    ),
    "technical_details": (
        "From a technical perspective, {model} utilizes a {architecture} with "
        "sophisticated attention mechanisms. It was {training}, which gives it "
        "the ability to understand complex patterns and relationships in text. "
        "The model processes {topic} by breaking down inputs into tokens and "
        "analyzing their relationships across multiple layers of the network."
    ),
    "advantages": (
        "The key strengths of {model} include its {strength} when handling {topic}. "
        "It excels at maintaining context over longer sequences and demonstrates "
        "strong performance on complex reasoning tasks. The model can generate "
        "coherent and contextually appropriate responses even for nuanced queries "
        "about {topic}."
    ),
    "limitations": (
        "Despite its capabilities, {model} has some limitations, including {weakness}. "
        "When dealing with {topic}, it may occasionally generate plausible-sounding "
        "but factually incorrect information. The model also inherits biases present "
        "in its training data and requires careful prompt engineering to achieve "
        "optimal results."
    ),
    "model_architecture": (
        "{model} features a {architecture} with multiple transformer layers and "
        "attention heads. Its architecture enables efficient processing of {topic} "
        "by utilizing self-attention mechanisms that capture relationships between "
        "tokens in the input sequence. The model was designed with a focus on "
        "balancing performance and computational efficiency."
    ),
    "training_data": (
        "The training process for {model} involved {training}. This approach "
        "allowed the model to develop a robust understanding of {topic} and related "
        "concepts. The training dataset encompasses a wide range of text styles, "
        "domains, and subjects, giving the model broad knowledge coverage."
    ),
    "strengths_weaknesses": (
        "{model} demonstrates significant {strength} when analyzing {topic}. "
        "It processes complex requests with high accuracy and contextual understanding. "
        "However, its {weakness} can manifest when dealing with ambiguous instructions "
        "or niche domains within {topic} that were underrepresented in its training data."
    ),
    "use_cases": (
        "{model} is particularly well-suited for applications involving {topic}, "
        "including content generation, data analysis, and decision support. "
        "Organizations have successfully deployed it for customer service automation, "
        "research assistance, and creative writing tasks. Its versatility makes it "
        "valuable across numerous industries and use cases."
    ),
}


def get_section_template(section: str) -> str:
    """
    Get the template for a specific section.

    Retrieves the appropriate template string for the requested section,
    normalizing the section name to a standard format. If no specific
    template exists, returns a generic template.

    Args:
        section: The section name to get a template for

    Returns:
        str: Template string with placeholders for customization

    Examples:
        >>> template = get_section_template("technical_details")
        >>> template.format(model="GPT-4", architecture="transformer", training="trained on diverse data", topic="code generation")
    """
    # Convert section name to standard format (snake_case)
    section_key: str = section.lower().replace(" ", "_")

    # Return the appropriate template or a generic one if not found
    return SECTION_TEMPLATES.get(
        section_key,
        (
            "Information about {model}'s {section} regarding {topic}. "
            "The model was developed by {company} in {year} and features "
            "a {architecture}."
        ).replace("{section}", section),
    )
