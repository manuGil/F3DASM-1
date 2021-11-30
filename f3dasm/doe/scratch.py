from f3dasm.doe.doevars import DoeVars
from f3dasm.doe.sampling import SalibSobol

# NOTES ON PACKAGING (Niket)
# separate high level requirements from python requirements
# what is need it to build a system,
# use py-project.doml  high level specifications, generate tar.bal: (executable)
## necessary to publish new versions of the package to pypi. 
## source -> platform independant, also how to build distributions for any platform
## wheel distributions -> platform dependant
# MANIFERST.in: pypi, declare packages that are need to be included, but need to be build in advance.
#  Look for references: https://packaging.python.org/tutorials/packaging-projects/#

# generate tar ball, then upload to pypi
# python setup.py 

# Always test in diferent python environment.
# look for test enve in doc as testpypi. 

# check list
# check if you can install somewhere else
# if everything is packed
# some things might not be available in testpypi
# mention dependencies for fenics
# mention docker option in installation instructions
