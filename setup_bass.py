from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='BASS',
      ext_modules=cythonize("bass.pyx",  compiler_directives={'language_level' : "3"},include_path=[np.get_include()]))
