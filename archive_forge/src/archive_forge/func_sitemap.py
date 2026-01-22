import datetime
from dataclasses import dataclass
from functools import wraps
from django.contrib.sites.shortcuts import get_current_site
from django.core.paginator import EmptyPage, PageNotAnInteger
from django.http import Http404
from django.template.response import TemplateResponse
from django.urls import reverse
from django.utils import timezone
from django.utils.http import http_date
@x_robots_tag
def sitemap(request, sitemaps, section=None, template_name='sitemap.xml', content_type='application/xml'):
    req_protocol = request.scheme
    req_site = get_current_site(request)
    if section is not None:
        if section not in sitemaps:
            raise Http404('No sitemap available for section: %r' % section)
        maps = [sitemaps[section]]
    else:
        maps = sitemaps.values()
    page = request.GET.get('p', 1)
    lastmod = None
    all_sites_lastmod = True
    urls = []
    for site in maps:
        try:
            if callable(site):
                site = site()
            urls.extend(site.get_urls(page=page, site=req_site, protocol=req_protocol))
            if all_sites_lastmod:
                site_lastmod = getattr(site, 'latest_lastmod', None)
                if site_lastmod is not None:
                    lastmod = _get_latest_lastmod(lastmod, site_lastmod)
                else:
                    all_sites_lastmod = False
        except EmptyPage:
            raise Http404('Page %s empty' % page)
        except PageNotAnInteger:
            raise Http404("No page '%s'" % page)
    if all_sites_lastmod:
        headers = {'Last-Modified': http_date(lastmod.timestamp())} if lastmod else None
    else:
        headers = None
    return TemplateResponse(request, template_name, {'urlset': urls}, content_type=content_type, headers=headers)