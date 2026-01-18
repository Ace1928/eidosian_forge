#!/bin/bash
# filepath: /home/lloyd/eidosian_forge/type_forge/docs/build_docs.sh

# Eidosian Type Forge Documentation Builder
# Precise, elegant documentation with modern components

set -e  # Exit on error for immediate feedback

echo "🔮 Building Eidosian Type Forge documentation with latest components"

# First ensure assets exist
if [ ! -f "_static/logo.png" ] || [ ! -f "_static/favicon.ico" ]; then
    echo "🎨 Creating essential assets"
    bash create_assets.sh
fi

# Apply precision fix for MyST footnote compatibility
export PYTHONNOUSERSITE=1
export PYTHONWARNINGS="ignore:Role 'sub-ref' is already registered:UserWarning"

# Clean previous build for pristine state
echo "🧹 Cleaning previous build artifacts"
rm -rf _build

# First create necessary directories if they don't exist
mkdir -p _templates/autoapi
mkdir -p _static

# Create custom template for handling typevar declarations
mkdir -p _templates/autoapi/python
cat > _templates/autoapi/python/class.rst << EOL
{% if obj.display %}
{% set typevar_names = [] %}
{% for base in obj.bases %}
    {% if 'typing.TypeVar' in base or 'TypeVar' in base %}
        {% set typevar_name = obj.name %}
        {% set typevar_names = typevar_names + [typevar_name] %}
    {% endif %}
{% endfor %}

{% if typevar_names %}
.. py:class:: {{ obj.name }}
   :module: {{ obj.module }}

   {{ obj.docstring|indent(3) }}
{% else %}
.. py:class:: {{ obj.name }}{% if obj.args %}({{ obj.args }}){% endif %}
   :module: {{ obj.module }}

   {{ obj.docstring|indent(3) }}

{% endif %}
{% endif %}
EOL

# Create module template to ensure complete template coverage
cat > _templates/autoapi/python/module.rst << EOL
{% if obj.display %}
.. py:module:: {{ obj.name }}

{% if obj.docstring %}
{{ obj.docstring|indent(3) }}
{% endif %}

{% if obj.children %}
{% set visible_children = obj.children|selectattr("display")|list %}
{% if visible_children %}
{{ obj.type|title }} Contents
{{ "-" * obj.type|length }}---------

{% for child in visible_children %}
{{ child.rendered|indent(3) }}
{% endfor %}
{% endif %}
{% endif %}
{% endif %}
EOL

# Check if custom CSS file exists, create if not
if [ ! -f "_static/custom.css" ]; then
    echo "📝 Creating custom CSS file"
    cat > _static/custom.css <<EOL
/* Custom styling for Type Forge documentation */
.wy-nav-content {
    max-width: 1200px;
}

/* Better code block styling */
.highlight {
    background: #f8f8f8;
    border-radius: 4px;
}

/* Styling for admonitions */
.admonition {
    border-radius: 4px;
}
EOL
fi

# Build with explicit compatibility options
echo "⚡ Building documentation with elegant conflict resolution"
sphinx-build -b html \
  -D suppress_warnings=myst.footnote,ref.python,toc.not_included,autoapi.python_import_resolution \
  -D warning_is_error=False \
  -W --keep-going \
  -n \
  . _build/html || true  # Don't fail on warnings

# Verify build success
if [ -f _build/html/index.html ]; then
    echo "✅ Documentation built successfully"
    echo "📚 View your documentation at: file://$(pwd)/_build/html/index.html"
    echo "🌐 Or serve with: python -m http.server 8000 --directory _build/html"
    exit 0
else
    echo "❌ Documentation build failed"
    exit 1
fi
