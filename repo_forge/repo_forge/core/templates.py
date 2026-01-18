"""
Template system for generating file content.

This module provides utilities for rendering templates with variables,
ensuring consistent style and content across the repository.
"""
from string import Template
from typing import Dict, Any, Optional
import re
from datetime import datetime


class EidosianTemplate(Template):
    """Enhanced template class with Eidosian principles applied."""
    
    def substitute(self, mapping: Dict[str, Any], **kws: Any) -> str:
        """
        Perform template substitution with enhanced error handling.
        
        Args:
            mapping: Dictionary of variable mappings
            **kws: Keyword arguments for variable mappings
            
        Returns:
            Rendered template string
        """
        # Merge mappings
        merged_mapping = {**mapping, **kws}
        
        # Add default values
        if 'current_date' not in merged_mapping:
            merged_mapping['current_date'] = datetime.now().strftime('%Y-%m-%d')
            
        if 'current_year' not in merged_mapping:
            merged_mapping['current_year'] = datetime.now().strftime('%Y')
        
        try:
            # Pre-process template for malformed placeholders
            template_str = self.template
            # Convert problematic ${var} syntax to $var if needed
            template_str = re.sub(r'\$\{([^}]+)\}', r'$\1', template_str)
            
            # Create a new template with the fixed string
            fixed_template = Template(template_str)
            return fixed_template.substitute(merged_mapping)
        except KeyError as e:
            missing_key = str(e).strip("'")
            # Provide a more helpful error message
            raise KeyError(
                f"Template is missing required variable '{missing_key}'. "
                f"Available variables: {', '.join(sorted(merged_mapping.keys()))}"
            ) from e
        except ValueError:
            # Instantly fall back to safe_substitute for any placeholder errors
            return self.safe_substitute(merged_mapping)
        except Exception:
            # Ultimate fallback for resilience
            return self.safe_substitute(merged_mapping)

    def safe_substitute(self, mapping: Dict[str, Any], **kws: Any) -> str:
        """
        Perform template substitution with fallback for missing variables.
        
        This enhanced version adds default date values as in the substitute method.
        
        Args:
            mapping: Dictionary of variable mappings
            **kws: Keyword arguments for variable mappings
            
        Returns:
            Rendered template string with untouched missing variables
        """
        # Merge mappings
        merged_mapping = {**mapping, **kws}
        
        # Add default values
        if 'current_date' not in merged_mapping:
            merged_mapping['current_date'] = datetime.now().strftime('%Y-%m-%d')
            
        if 'current_year' not in merged_mapping:
            merged_mapping['current_year'] = datetime.now().strftime('%Y')
        
        # Attempt pre-processing for improved consistency
        try:
            template_str = self.template
            # Convert problematic ${var} syntax to $var if needed
            template_str = re.sub(r'\$\{([^}]+)\}', r'$\1', template_str)
            
            # Create a new template with the fixed string
            fixed_template = Template(template_str)
            return fixed_template.safe_substitute(merged_mapping)
        except:
            # Fallback to original behavior
            return super().safe_substitute(merged_mapping)


def render_template(template_str: str, variables: Dict[str, Any], safe: bool = False) -> str:
    """
    Render a template string with provided variables.
    
    Args:
        template_str: Template string with $variable placeholders
        variables: Dictionary of variable values
        safe: If True, use safe_substitute to ignore missing variables
        
    Returns:
        Rendered template string
    """
    # Critical fix: Add datetime module to variables for legacy template support
    variables_with_defaults = {**variables}
    
    # Ensure datetime is available as a placeholder variable
    if 'datetime' not in variables_with_defaults:
        variables_with_defaults['datetime'] = datetime
        
    template = EidosianTemplate(template_str)
    try:
        if not safe:
            return template.substitute(variables_with_defaults)
        return template.safe_substitute(variables_with_defaults)
    except Exception as e:
        # Fallback mechanism for error recovery
        print(f"Warning: Template rendering error, using safe mode: {str(e)}")
        return template.safe_substitute(variables_with_defaults)


def render_comment_block(content: str, style: str = 'python') -> str:
    """
    Wrap content in a language-appropriate comment block.
    
    Args:
        content: Content to wrap in comments
        style: Programming language style ('python', 'bash', etc.)
        
    Returns:
        Comment-wrapped content
    """
    comment_styles = {
        'python': {'start': '"""', 'end': '"""', 'line': ''},
        'bash': {'start': '', 'end': '', 'line': '# '},
        'javascript': {'start': '/**', 'end': ' */', 'line': ' * '},
        'go': {'start': '/*', 'end': ' */', 'line': ' * '},
        'rust': {'start': '//!', 'end': '', 'line': '//! '},
        'html': {'start': '<!--', 'end': '-->', 'line': ''},
        'css': {'start': '/*', 'end': ' */', 'line': ' * '},
    }
    
    style_info = comment_styles.get(style.lower(), comment_styles['python'])
    
    if style_info['start'] and style_info['end']:
        lines = content.split('\n')
        if style_info['line']:
            lines = [f"{style_info['line']}{line}" if line else style_info['line'].rstrip() for line in lines]
        return f"{style_info['start']}\n{chr(10).join(lines)}\n{style_info['end']}"
    elif style_info['line']:
        lines = content.split('\n')
        return '\n'.join(f"{style_info['line']}{line}" for line in lines)
    else:
        return content