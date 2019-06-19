# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 21:36:07 2016

@author: michaellynn
"""

from setuptools import setup

setup(name='silent_synapse_mle',
      version='0.1',
      description='Performs maximum likelihood estimation on sets of FRA data'
      url='none',
      author='Michael Lynn',
      author_email='micllynn@gmail.com',
      license='MIT',
      packages=['silent_synapse_mle'],
      install_requires = ['numpy', 'scipy', 'matplotlib'],
      zip_safe = False)
