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
vars = {'Fs': SalibSobol(2, {'F11':[-0.15, 1], 'F12':[-0.1,0.15], 'F22':[-0.15, 1]}),
        'R': SalibSobol(3, {'radius': [0.3, 0.5]}),
            'particle': ({ 'NeoHookean':
                            {'E': SalibSobol(3,{'e':[0.3, 0.5]}), 
                            'nu': 0.4 
                            } },
                        {'SaintVenant':  
                            {'E': [5, 200, 300],
                            'nu': SalibSobol(5,{'e':[0.3, 0.5]})
                            } },
                            ),
            'matrix': {  
                'name': 'SaintVenant',  
                'E': SalibSobol(3,{'e':[0.3, 0.5]}),
                'nu': 0.3
            },
            'Vf': 0.3,
            'Lc': 4,
            'geometry': 'circle'
            }

# vars = {'Fs': SalibSobol(2, {'F11':[-0.15, 1], 'F12':[-0.1,0.15], 'F22':[-0.15, 1]}), 'R': [0.3, 0.5] }


doe = DoeVars(vars)



print('DoEVars definition:')
# print(doe.sampling_vars)

print('\n DoEVars summary information:')
# print(doe.info())

# # Compute sampling and combinations
doe.do_sampling()

# print('\n Pandas dataframe with compbined-sampled values:')
# # print(doe.data)
