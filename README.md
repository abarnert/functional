functional
==========
Collin Winter's functional module seems to be dead (no updates since
2006, and the homepage at http://oakwinter.com/code/functional/ is
long gone), and does not work right with Python 3.x or even completely
with 2.7 (despite the 2.7 docs having a link to the dead homepage).

This is almost a trivial port using 2to3, but the test suite needed
some tweaking, so it's been split into two suites: tests should run
on 2.3-2.6, tests3 on 2.6+ (including 3.x).

Introduction
------------
functional provides Python users with numerous tools common in
functional programming, such as foldl, foldr, flip, as well as
mechanisms for partial function application and function composition.
functional also includes sane versions of the Python builtins map()
and filter(), written without the weird semantics of the builtin
versions.

Installation
------------
Directly from PyPI:     
> Not yet...

Via pip:

    pip install git+https://github.com/abarnert/functional.git

From source:
  1. Untar the .tar.gz file
  2. cd into the resulting directory
  3. python setup.py install

From a Python egg:
> Not yet...

Documentation
-------------
For now, refer to the old website via the Wayback Machine, at:

> http://web.archive.org/web/20101018142225/http://oakwinter.com/code/functional/

Copyright/License
-----------------
The initial commit of this project is an exact copy of the last
version of Collin Winter's module, extracted from
[PyPI](http://pypi.python.org/packages/source/f/functional/functional-Py-0.7.0.tar.gz#md5=8fbdc43b8ba5200e95c2b028f3d5569e)
and committed to
[github](https://github.com/abarnert/functional/commit/97dc01ff9d07bccde2f18445dcb7d1592e92dcae)
with no changes except the addition of a generic README.md file.

The license for version 0.7.0 is, according to the PKG-INFO, the
Python Software Foundation License.

The original documentation says only "See the copyright header in
functionalmodule.c", a file which doesn't exist. Likewise, the
documentation for the C version of this library said "See the
copyright header in functional/__init__.py", which does not exist in
that version. (Also, neither version is part of Python, but they
are copyright by the PSF.)