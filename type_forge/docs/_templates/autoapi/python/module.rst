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
