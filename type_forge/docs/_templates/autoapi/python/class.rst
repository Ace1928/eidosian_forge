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
