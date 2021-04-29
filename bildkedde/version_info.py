__all__ = [
            '__name__',
            '__version__',
            '__description__',
            '__author__',
            '__email__',
            '__license__',
            '__url__',
            '__download_url__',
            '__maintainer__',
            '__maintainer_email__',
            '__keywords__',
            '__python_requires__',
            '__platforms__',
            '__classifiers__',
            '__credits__',
            '__status__',
            '__copyright__',
            ]


__name__ = 'bildkedde'
__version__ = '0.0.1-dev'
__description__ = 'Lightweight sensor modeling package'

__author__ = 'Ryan Hartzell'
__email__ = "bildkedde@gmail.com"
__license__ = "MIT"
__url__ = "https://ryanhartzell.github.io/Bildkedde"
__download_url__ = 'https://github.com/RyanHartzell/Bildkedde'
__maintainer__ = "Ryan Hartzell"
__maintainer_email__ = "bildkedde@gmail.com"
__keywords__ = 'imaging-science sensor-model image-chain optics radiometry'
__python_requires__ = '>=3.7.*'
__platforms__ = ["Windows", "Linux", "Mac OS-X", "Unix"]
__classifiers__ = [
    'Development Status :: 1 - Planning',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Education',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Image Processing',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: Implementation :: CPython',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Natural Language :: English',
    ]

__credits__ = ["Ryan Hartzell",]
__status__ = "Experimental"

import datetime
__copyright__ = f"Copyright (c) 2021-{datetime.datetime.now().year} " + __author__
del datetime