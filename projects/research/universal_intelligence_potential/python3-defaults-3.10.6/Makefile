#!/usr/bin/make -f
INSTALL ?= install
PREFIX ?= /usr/local
MANPAGES ?= py3compile.1 py3clean.1
VERSION=$(shell dpkg-parsechangelog | sed -rne 's,^Version: (.+),\1,p')

clean:
	find . -name '*.py[co]' -delete
	find . -name __pycache__ -type d | xargs rm -rf
	rm -f .coverage $(MANPAGES)

install-dev:
	$(INSTALL) -m 755 -d $(DESTDIR)$(PREFIX)/bin \
		$(DESTDIR)$(PREFIX)/share/python3/runtime.d
	$(INSTALL) -m 755 runtime.d/* $(DESTDIR)$(PREFIX)/share/python3/runtime.d/

install-runtime:
	$(INSTALL) -m 755 -d $(DESTDIR)$(PREFIX)/share/python3/debpython $(DESTDIR)$(PREFIX)/bin
	$(INSTALL) -m 644 debpython/*.py $(DESTDIR)$(PREFIX)/share/python3/debpython/
	$(INSTALL) -m 755 py3compile $(DESTDIR)$(PREFIX)/bin/
	sed -i -e 's/DEVELV/$(VERSION)/' $(DESTDIR)$(PREFIX)/bin/py3compile
	$(INSTALL) -m 755 py3clean $(DESTDIR)$(PREFIX)/bin/
	sed -i -e 's/DEVELV/$(VERSION)/' $(DESTDIR)$(PREFIX)/bin/py3clean

install: install-dev install-runtime

%.1: %.rst
	rst2man $< > $@

manpages: $(MANPAGES)

sbuild:
	sbuild --debbuildopts -I

# TESTS
tests:
	nosetests3 --with-doctest --with-coverage

check_versions:
	@PYTHONPATH=. set -e; \
	DEFAULT=`python3 -c 'import debpython.version as v; print(v.vrepr(v.DEFAULT))'`;\
	SUPPORTED=`python3 -c 'import debpython.version as v; print(" ".join(sorted(v.vrepr(v.SUPPORTED))))'`;\
	DEB_DEFAULT=`sed -rn 's,^default-version = python([0.9.]*),\1,p' debian/debian_defaults`;\
	DEB_SUPPORTED=`sed -rn 's|^supported-versions = (.*)|\1|p' debian/debian_defaults | sed 's/python//g;s/,//g'`;\
	[ "$$DEFAULT" = "$$DEB_DEFAULT" ] || \
	(echo 'Please update DEFAULT in debpython/version.py' >/dev/stderr; false);\
	[ "$$SUPPORTED" = "$$DEB_SUPPORTED" ] || \
	(echo 'Please update SUPPORTED in debpython/version.py' >/dev/stderr; false)

.PHONY: clean tests test% check_versions
