# -*- coding: utf-8 -*-
"""
@author: michaellynn
"""

from setuptools import setup

setup(name='silentmle',
      version='1.0',
      description='Performs maximum likelihood estimation on sets of FRA data',
      url='https://github.com/micllynn/SilentMLE',
      author='Michael B. Lynn',
      author_email='micllynn@gmail.com',
      license='MIT',
      packages=['silentmle'],
      python_requires='>=3.6',
      install_requires=['numpy', 'scipy', 'matplotlib', 'h5py', 'seaborn'],
      zip_safe=False)
