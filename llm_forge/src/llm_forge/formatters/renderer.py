"""
Output renderer for model comparison results.

This module transforms ModelResponse objects into formatted text,
supporting multiple output formats including markdown, HTML, and
plain text. Each renderer ensures consistent, well-organized presentation
of model comparisons.
"""

from typing import Callable, Dict, Final, List, Literal, TypedDict

from llm_forge.logging_config import configure_logging
from llm_forge.type_definitions import ModelResponse

# Configure renderer-specific logger
logger = configure_logging()

# Format type definition
FormatType = Literal["markdown", "html", "text"]

# Type for renderer functions
RendererFunc = Callable[[ModelResponse], str]


# Renderer mapping type for improved type safety
class RendererMap(TypedDict):
    """Type definition for renderer function mapping."""

    markdown: RendererFunc
    html: RendererFunc
    text: RendererFunc


def render_output(
    comparison_data: ModelResponse, output_format: FormatType = "markdown"
) -> str:
    """
    Render comparison data in the specified output format.

    Takes structured model comparison data and transforms it into
    a formatted string suitable for the requested output medium.

    Args:
        comparison_data: Structured model comparison data
        output_format: Desired output format (markdown, html, or text)

    Returns:
        str: Formatted string representation of the comparison data

    Raises:
        ValueError: If the requested format is not supported

    Examples:
        >>> data = {"topic": "AI Models", "models": {...}}
        >>> markdown = render_output(data, "markdown")
    """
    # Map format types to their respective rendering functions
    format_renderers: Final[RendererMap] = {
        "markdown": _render_markdown,
        "html": _render_html,
        "text": _render_text,
    }

    # Validate the requested format
    if output_format not in format_renderers:
        valid_formats: str = ", ".join(format_renderers.keys())
        raise ValueError(
            f"Unsupported output format: {output_format}. Valid formats: {valid_formats}"
        )

    # Get and execute the appropriate renderer
    renderer: RendererFunc = format_renderers[output_format]
    logger.info(f"Rendering comparison in {output_format} format")

    return renderer(comparison_data)


def _render_markdown(comparison_data: ModelResponse) -> str:
    """
    Render comparison data as Markdown.

    Creates a well-structured Markdown document with headings, lists,
    and formatted text for easy reading and further processing.

    Args:
        comparison_data: The model comparison data to render

    Returns:
        str: Markdown string representing the comparison

    Note:
        Markdown output includes a table of contents and comparison summary
    """
    topic: str = comparison_data["topic"]
    models: Dict[str, Dict[str, str]] = comparison_data["models"]

    # Initialize output with title
    output: List[str] = [f"# Model Comparison: {topic}\n"]

    # Add table of contents
    output.append("## Table of Contents\n")
    for i, model_name in enumerate(models.keys()):
        output.append(
            f"{i+1}. [{model_name.upper()}](#{model_name.lower().replace(' ', '-')})\n"
        )
    output.append("\n---\n")

    # Add model sections
    for model_name, sections in models.items():
        output.append(f"## {model_name.upper()}\n")

        # Add each content section
        for section_name, content in sections.items():
            formatted_section: str = section_name.replace("_", " ").title()
            output.append(f"### {formatted_section}\n")
            output.append(f"{content}\n\n")

        output.append("---\n")

    # Add comparison summary
    output.append("## Summary Comparison\n")
    output.append("| Model | Key Strengths | Notable Limitations |\n")
    output.append("| ----- | ------------- | ------------------ |\n")

    for model_name, sections in models.items():
        strengths: str = "N/A"
        limitations: str = "N/A"

        # Extract brief versions of strengths and limitations if available
        if "advantages" in sections:
            strengths = sections["advantages"].split(". ")[0]
        if "limitations" in sections:
            limitations = sections["limitations"].split(". ")[0]

        output.append(f"| **{model_name.upper()}** | {strengths} | {limitations} |\n")

    return "".join(output)


def _render_html(comparison_data: ModelResponse) -> str:
    """
    Render comparison data as HTML.

    Creates a clean, structured HTML document with appropriate
    semantic elements and minimal styling for web presentation.

    Args:
        comparison_data: The model comparison data to render

    Returns:
        str: HTML string representing the comparison

    Note:
        Includes basic CSS styling and a responsive layout
    """
    topic: str = comparison_data["topic"]
    models: Dict[str, Dict[str, str]] = comparison_data["models"]

    # Create HTML structure
    html: List[str] = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"    <title>Model Comparison: {topic}</title>",
        "    <style>",
        "        body { font-family: system-ui, sans-serif; line-height: 1.5; max-width: 900px; margin: 0 auto; padding: 1rem; }",
        "        h1, h2, h3 { margin-top: 1.5em; }",
        "        .model-section { border: 1px solid #eee; border-radius: 5px; padding: 1em; margin: 1em 0; }",
        "        .comparison-table { width: 100%; border-collapse: collapse; }",
        "        .comparison-table th, .comparison-table td { border: 1px solid #ddd; padding: 0.5em; text-align: left; }",
        "        .comparison-table th { background-color: #f5f5f5; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>Model Comparison: {topic}</h1>",
        "",
        "    <h2>Table of Contents</h2>",
        "    <ul>",
    ]

    # Add table of contents
    for model_name in models.keys():
        model_id: str = model_name.lower().replace(" ", "-")
        html.append(f"        <li><a href='#{model_id}'>{model_name.upper()}</a></li>")

    html.append("    </ul>")
    html.append("    <hr>")

    # Add model sections
    for model_name, sections in models.items():
        model_id: str = model_name.lower().replace(" ", "-")
        html.append(f"    <div id='{model_id}' class='model-section'>")
        html.append(f"        <h2>{model_name.upper()}</h2>")

        # Add each content section
        for section_name, content in sections.items():
            formatted_section: str = section_name.replace("_", " ").title()
            html.append(f"        <h3>{formatted_section}</h3>")
            # Format paragraphs correctly with <p> tags
            paragraphs: List[str] = content.split("\n\n")
            for para in paragraphs:
                html.append(f"        <p>{para}</p>")

        html.append("    </div>")

    # Add comparison table
    html.append("    <h2>Summary Comparison</h2>")
    html.append("    <table class='comparison-table'>")
    html.append(
        "        <tr><th>Model</th><th>Key Strengths</th><th>Notable Limitations</th></tr>"
    )

    for model_name, sections in models.items():
        strengths: str = "N/A"
        limitations: str = "N/A"

        # Extract brief versions of strengths and limitations if available
        if "advantages" in sections:
            strengths = sections["advantages"].split(". ")[0]
        if "limitations" in sections:
            limitations = sections["limitations"].split(". ")[0]

        html.append(
            f"        <tr><td><strong>{model_name.upper()}</strong></td><td>{strengths}</td><td>{limitations}</td></tr>"
        )

    html.append("    </table>")
    html.append("</body>")
    html.append("</html>")

    return "\n".join(html)


def _render_text(comparison_data: ModelResponse) -> str:
    """
    Render comparison data as plain text.

    Creates a clean, structured plain text document with clear
    section delineation and consistent formatting.

    Args:
        comparison_data: The model comparison data to render

    Returns:
        str: Plain text string representing the comparison

    Note:
        Uses ASCII-based formatting with horizontal rules and aligned columns
    """
    topic: str = comparison_data["topic"]
    models: Dict[str, Dict[str, str]] = comparison_data["models"]

    # Calculate width for horizontal rules
    hr_width: int = 80
    hr: str = "=" * hr_width

    # Initialize output with title
    output: List[str] = [
        hr,
        f"MODEL COMPARISON: {topic.upper()}".center(hr_width),
        hr,
        "",
    ]

    # Add table of contents
    output.append("CONTENTS:")
    for i, model_name in enumerate(models.keys()):
        output.append(f"{i+1}. {model_name.upper()}")
    output.append("")
    output.append("-" * hr_width)
    output.append("")

    # Add model sections
    for model_name, sections in models.items():
        output.append(f"{model_name.upper()}".center(hr_width))
        output.append("-" * hr_width)

        # Add each content section
        for section_name, content in sections.items():
            formatted_section: str = section_name.replace("_", " ").title()
            output.append(f"{formatted_section}:")
            output.append("")
            output.append(content)
            output.append("")

        output.append(hr)
        output.append("")

    # Add comparison summary
    output.append("SUMMARY COMPARISON:")
    output.append("")

    # Calculate column widths for table
    model_width: int = max(len(model_name.upper()) for model_name in models.keys()) + 2
    column_width: int = (hr_width - model_width - 3) // 2

    # Add table header
    output.append(
        f"{'MODEL'.ljust(model_width)} | {'KEY STRENGTHS'.ljust(column_width)} | {'NOTABLE LIMITATIONS'}"
    )
    output.append(f"{'-' * model_width}-+-{'-' * column_width}-+-{'-' * column_width}")

    # Add model rows
    for model_name, sections in models.items():
        strengths: str = "N/A"
        limitations: str = "N/A"

        # Extract brief versions of strengths and limitations if available
        if "advantages" in sections:
            strengths = sections["advantages"].split(". ")[0][:column_width]
        if "limitations" in sections:
            limitations = sections["limitations"].split(". ")[0][:column_width]

        output.append(
            f"{model_name.upper().ljust(model_width)} | {strengths.ljust(column_width)} | {limitations}"
        )

    return "\n".join(output)
