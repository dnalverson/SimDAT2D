# setup.py file for the TFData package
from setuptools import setup, find_packages

setup(
    name='thin-film-data',
    version='0.0.2',
    description='To cerate simulated 2D images of calibrant materials on single crystal substrates and integrate them with multiple integration patterns',
    long_description= 'This package is designed to create simulated 2D images of calibrant materials on different orientations of single crystal substrates and integrate them to get multiple integration patterns. The integration patterns can be used for further analysis to extract and separate the single crystal background and calibrant material of interest.',
    author='Danielle N. Alverson',
    author_email='dalverson@ufl.edu',
    packages=find_packages(),
    url='https://github.com/dnalverson/thinfilm-toy-data-tool/',
    install_requirements = ['numpy', 'matplotlib', 'scipy', 'pyFAI', 'pyFAI.gui', 'pyFAI.detectors', 'pyFAI.azimuthalIntegrator']
)