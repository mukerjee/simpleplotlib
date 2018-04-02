from setuptools import setup

setup(
    name='simpleplotlib',
    packages=['simpleplotlib'],
    version='0.17',
    description='A matplotlib wrapper focused on beauty and simplicity',
    author='Matthew K. Mukerjee',
    author_email='Matthew.Mukerjee@gmail.com',
    url='https://github.com/mukerjee/simpleplotlib',
    download_url='https://github.com/mukerjee/simpleplotlib/tarball/0.17',
    license='MIT License',
    keywords=['matplotlib', 'plots', 'beauty', 'simplicity'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Topic :: Scientific/Engineering'
    ],
    install_requires=[
        'matplotlib>=2.0.0',
        'numpy>=1.14.0',
        'dotmap',
    ]
)
