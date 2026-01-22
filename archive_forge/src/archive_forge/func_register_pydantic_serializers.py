from pkg_resources import packaging
def register_pydantic_serializers(serialization_context):
    if not PYDANTIC_INSTALLED:
        return
    if IS_PYDANTIC_2:
        from pydantic.v1.fields import ModelField
    else:
        from pydantic.fields import ModelField
    serialization_context._register_cloudpickle_serializer(ModelField, custom_serializer=lambda o: {'name': o.name, 'type_': o.outer_type_, 'class_validators': o.class_validators, 'model_config': o.model_config, 'default': o.default, 'default_factory': o.default_factory, 'required': o.required, 'alias': o.alias, 'field_info': o.field_info}, custom_deserializer=lambda kwargs: ModelField(**kwargs))