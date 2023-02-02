# setup.py file for the TFData package
from setuptools import setup, find_packages

setup(
    name='TFData',
    version='0.0.1',
    description='To cerate simulated 2D images of calibrant materials on single crystal substrates and integrate them with multiple integration patterns',
    author='Danielle N. Alverson',
    author_email='dalverson@ufl.edu',
    packages=find_packages(),
    url='https://github.com/dnalverson/thinfilm-toy-data-tool/',
    install_requirements = ['numpy', 'matplotlib', 'scipy', 'pyFAI', 'pyFAI.gui', 'pyFAI.detectors', 'pyFAI.azimuthalIntegrator']
)