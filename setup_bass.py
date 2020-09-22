from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(name='BASS',
      ext_modules=cythonize(Extension("bass",sources=["bass.pyx"],include_dirs=[np.get_include()]), compiler_directives={'language_level' : "3"})
      )
