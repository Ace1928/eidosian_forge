from django.apps import apps
from django.contrib.contenttypes.models import ContentType
from django.contrib.sites.shortcuts import get_current_site
from django.core.exceptions import ObjectDoesNotExist
from django.http import Http404, HttpResponseRedirect
from django.utils.translation import gettext as _
def shortcut(request, content_type_id, object_id):
    """
    Redirect to an object's page based on a content-type ID and an object ID.
    """
    try:
        content_type = ContentType.objects.get(pk=content_type_id)
        if not content_type.model_class():
            raise Http404(_('Content type %(ct_id)s object has no associated model') % {'ct_id': content_type_id})
        obj = content_type.get_object_for_this_type(pk=object_id)
    except (ObjectDoesNotExist, ValueError):
        raise Http404(_('Content type %(ct_id)s object %(obj_id)s doesn’t exist') % {'ct_id': content_type_id, 'obj_id': object_id})
    try:
        get_absolute_url = obj.get_absolute_url
    except AttributeError:
        raise Http404(_('%(ct_name)s objects don’t have a get_absolute_url() method') % {'ct_name': content_type.name})
    absurl = get_absolute_url()
    if absurl.startswith(('http://', 'https://', '//')):
        return HttpResponseRedirect(absurl)
    try:
        object_domain = get_current_site(request).domain
    except ObjectDoesNotExist:
        object_domain = None
    if apps.is_installed('django.contrib.sites'):
        Site = apps.get_model('sites.Site')
        opts = obj._meta
        for field in opts.many_to_many:
            if field.remote_field.model is Site:
                site_qs = getattr(obj, field.name).all()
                if object_domain and site_qs.filter(domain=object_domain).exists():
                    break
                site = site_qs.first()
                if site:
                    object_domain = site.domain
                    break
        else:
            for field in obj._meta.fields:
                if field.remote_field and field.remote_field.model is Site:
                    try:
                        site = getattr(obj, field.name)
                    except Site.DoesNotExist:
                        continue
                    if site is not None:
                        object_domain = site.domain
                        break
    if object_domain is not None:
        protocol = request.scheme
        return HttpResponseRedirect('%s://%s%s' % (protocol, object_domain, absurl))
    else:
        return HttpResponseRedirect(absurl)