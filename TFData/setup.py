# setup.py file for the TFData package
from setuptools import setup, find_packages

setup(
    name='TFData',
    version='0.0.1',
    description='To cerate simulated 2D images of calibrant materials on single crystal substrates and integrate them with multiple integration patterns',
    author='Danielle N. Alverson',
    author_email='dalverson@ufl.edu',
    url='https://github.com/dnalverson/thinfilm-toy-data-tool/',
    packages=find_packages(),
    install_requires=[pyFAI, matplotlib, pandas, numpy, scipy],
)