from setuptools import find_packages, setup

setup(
    name='if-stitch',
    version='1.0',
    extras_require=dict(tests=['pytest']),
    url='https://github.com/marx-alex/if-stitch',
    license='MIT',
    author='Alexander Marx',
    author_email='',
    description='2D-Stitching for microscope images',
    classifiers=['Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Image Preprocessing :: Stitching',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8'],
    keywords='medicine microscopy stitching histology cells biology mosaic',
    install_requires=['numpy>=1.19.3',
                      'opencv-contrib-python>=4.4.0.46',
                      'imutils>=0.5.3']
)
