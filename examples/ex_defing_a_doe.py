"""
Example on how to define a DoE
This example will be moved to the documentation
"""

from f3dasm.doe.doevars import  DoeVars
from f3dasm.doe.sampling import SalibSobol

# cases:
# 1. var defines a constant -> no sampling, convert to data frame, combine with cartesian product
# 2. var defines a vector -> no sampling, convert to data frame, combine with cartesian product
# 3. var defines a sampling method with 1 or more sub vars -> apply sampling, convert to data frame using columns, combine with cartesian product
# 4. var defines a set (dict) of cases 1, 2 and 3. -> do 1, 2 or 3 for sampling and convertion
# 5. var defines a tupple containing 2 or more of cases 1-4 -> do sampling as on 1, 2,3 or 4, combine element of tupple with append


# define variables for the DoE as a dictionary, for example
vars = { # and operator
        'Fs': SalibSobol(2, {'F11':[-0.15, 1], 'F12':[-0.1,0.15], 'F22':[-0.15, 1]}),
        'R': SalibSobol(3, {'radius': [0.3, 0.5]}), # var as function
            'particle': ( # or operator
                        { 'NeoHookean': # operand 1 (item)
                            {'E0': SalibSobol(3,{'e':[0.3, 0.5]}), # and operator
                            'nu0': 0.4
                            } }, # TODO: fix issue when number of varialbes is more than 2
                        {'SaintVenant':   # operand 2 (item)
                            {'E1': [5, 200, 300], # vector
                            'nu1': SalibSobol(5,{'e':[0.3, 0.5]})
                            } },
                            ),
            'matrix': {  
                'name': 'SaintVenant',  # constant
                'E': SalibSobol(3,{'e':[0.3, 0.5]}),
                'nu': 0.3
            },
            'Vf': 0.3,
            'Lc': 4,
            'geometry': 'circle'
            }


# vars = {'Fs': SalibSobol(3, {'F11':[-0.15, 1], 'F12':[-0.1,0.15], 'F22':[-0.15, 1]}), 'R': 3 }


doe = DoeVars(vars)

print('DoEVars definition:')
# print(doe.sampling_vars)

print('\n DoEVars summary information:')
# print(doe.info())

# # Compute sampling and combinations
doe.do_sampling()

# doe.combine()
# print('\n Pandas dataframe with compbined-sampled values:')
# print(doe.data)

# doe.data