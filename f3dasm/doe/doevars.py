
#######################################################
# Data class for storing variables (parameters) of
# the design of experiments F3DASM                               #
#######################################################

from dataclasses import dataclass, field
import pandas as pd
from pandas.core.frame import DataFrame
from f3dasm.doe.sampling import SamplingMethod, SalibSobol
import copy
import numpy


def print_variables(dictionary:dict):
    """Print the top level elements in a dictionary"""

    keys = dictionary.keys()
    for k in keys:
        print(k,':', dictionary[k])

    return None


def classify_vars(doe_vars: dict):
    """
    Classify DoeVars  into fixed and sampling variables. 
    Sampling variables have values that are instance of SamplingMethod
    WARNING: only 3 levels of nesting are checked in the input.
    Args:
        doe_vars (dict): variables defining a design of experiment
    Returns:
        dictonary with list of variable-names
    """
    # expand dictionary
    df = pd.json_normalize(doe_vars)
    vars_level1 = df.to_dict(orient='records')[0]

    sampling_variables = [] # list of names
    fixed_variables =[]
    # loop over variables at the top level of serialization
    for key, value in vars_level1.items():
        # id to keep track of which variable belog to the same group
        # group ids will will start with f3dams# followed by the value group_id
        group_id = 0
        if isinstance(value, SamplingMethod):
            sampling_variables.append(key)

        # case for OR operator, 
        # variable grouped as a tupple are combined differently
        # then ungrouped variables, see create_combinations()
        elif isinstance(value, tuple):
            for grouped_var in value:
                g_df = pd.json_normalize(grouped_var)
                g_vars = g_df.to_dict(orient='records')[0]
                for g_key, g_value in g_vars.items():
                    if isinstance(g_value, SamplingMethod):
                        sampling_variables.append('f3dasm#'+str(group_id)+'.'+key+'.'+g_key)
        else:
            fixed_variables.append(key)    
            # continue
        group_id+=1
    # TODO: find a generic solution this function. So that it works for 
    # any level of nesting.
    return {'fixed': fixed_variables, 'sampling': sampling_variables}


def deserialize_dictionary(nested_dict: dict):
    """Deserialize a nested dictionary"""
    
    norm_ = pd.json_normalize(nested_dict)
    return norm_.to_dict(orient='records')[0]


def create_combinations(func, args):
        """wrapper for computing combinations of DoE variables"""
        columns = len(args)
        try: 
            result = func(*args)
        finally:
            return numpy.array(result).T.reshape(-1, columns)

def apply_sampling(sampling_var, column_names):
    """
    Applies sampling to varible with a sampling method and coverts result to data frame
    Args:
        sampling_var (dict): single-element dictionary which value is an instance of SamplingMethod.
        coumn_names: list of names for the columns of the data frame, must match number of colunms in sampling result.
    Returns: pandas data frame. 
    """
    samples = sampling_var.compute_sampling()

    #check number of columns
    if samples.shape[1] != len(column_names):
        raise ValueError('Number of columns in samples-array and column names do not match')
    data_frame = pd.DataFrame(data=samples, columns=column_names)
    return data_frame


@dataclass
class DoeVars:
    """Parameters for the design of experiments"""

    variables: dict
    sampling_vars: list = field(init=False)
    data: DataFrame = None

    def __post_init__(self):
        self.sampling_vars = find_sampling_vars(self.variables)

    def info(self):

        """ Overwrite print function"""

        print('-----------------------------------------------------')
        print('                       DOE VARIABLES                     ')
        print('-----------------------------------------------------')
        print_variables(self.variables)
        print('\n')

        return None

      
    def do_sampling(self) -> DataFrame:
        """Apply sampling method to sampling variables, combines sampled value and fixed-values,
        and produces a pandas data frame with all combinations.
        """
        # make copy of doe variables
        doe_vars = copy.deepcopy(self.variables)
        
        # print('sampling var:', self.sampling_vars)

        counter = 0 # counter for the case of the OR operator
        
        # collects data frames grouped by operator:
        # Data frames in the 'and' group will be combined using cartesian product
        # Data frames in the 'or' group will be combined by appeding columns and rows
        collector = {'and': [], 'or': []} 
        
        # ################################
        # Compute sampling for variable 
        # with a sampling method
        ##################################
        for var in self.sampling_vars:
            inner_vars = var.split('.') 
            # print('inner vars:', inner_vars)
            # CASES:
            # ##########################
            # AND GROUP
            # ##########################     
            if len(inner_vars) == 1:

                df1 =apply_sampling(doe_vars[var], list(doe_vars[var].sampling_ranges.keys()))

                collector['and'].append(df1)

                # print('df1', df1)

            ## sampling variable at a second level that DO NOT define a group 
            # (use OR operator). Groups contains an element which starts with 
            # the string 'f3dasm#'
            elif not "f3dasm#" in inner_vars[0]: 
                # print(inner_vars)
                if len(inner_vars) == 2:
                    df2 = apply_sampling(doe_vars[inner_vars[0]][inner_vars[1]], [ inner_vars[0]+'.'+inner_vars[1] ])

                    collector['and'].append(df2)
                    # print('df2', df2)

                elif len(inner_vars) == 3:
                    
                    df3 = apply_sampling(
                                    doe_vars[inner_vars[0]][inner_vars[1]][inner_vars[2]], 
                                    [ inner_vars[0]+'.'+inner_vars[1]+'.'+inner_vars[2]]
                                    )   
                    collector['and'].append(df3)                 
                    # print(df3)
                else:
                    raise ValueError("DoeVars definition contains too many nested elements. A max of 3 is allowed")

            # ##########################
            # OR GROUP
            # ##########################
            elif "f3dasm#" in inner_vars[0]:
                
                inner_v = copy.deepcopy(inner_vars)
                # remove group id
                inner_v.pop(0)
                
                # print('target', doe_vars[inner_v[0]] [counter] [inner_v[1]])
                
                # apply sampling to nested variable in the OR operator group
                if len(inner_v) == 1:
                    # Malformed tupple, containing one element
                    raise SyntaxError(f'Group defining OR operator requires at least two variables: {inner_v}')

                # Apply sampling based on the number of nested varibles for the 
                # OR operator
                elif len(inner_v) == 2:
                    df4 = apply_sampling(doe_vars[inner_v[0]][counter][inner_v[1]], [ inner_v[0]+'.'+inner_v[1] ])

                    collector['or'].append(df4)

                    # print('df4', df4)

                elif len(inner_v) == 3:
                            
                    df5 = apply_sampling(
                                    doe_vars[inner_v[0]][counter][inner_v[1]][inner_v[2]], 
                                    [ inner_v[0]+'.'+inner_v[1]+'.'+inner_v[2]]
                                    )  

                    collector['or'].append(df5)                  
                    # print(df5)
                else:
                    raise ValueError("DoeVars definition contains too many nested elements. A max of 3 is allowed")

                # When group ids don't match, restart the counter
                if inner_vars[0] != "f3dasm#" + str(counter):
                    counter = 0
                else:
                    counter += 1

            else:
                raise ValueError('The required operation is not implemented')

        # sampled_values = list( deserialize_dictionary(doe_vars).values() )
        # combinations = create_combinations(numpy.meshgrid, sampled_values)

        # # dataframe
        # _columns =list( deserialize_dictionary(doe_vars).keys() )
        # self.data = pd.DataFrame(combinations,columns=_columns)
        # return self.data
        return collector

    def save(self,filename):

        """ Save doe-vars as Pandas data frame in a pickle file
        Args:
            filename (string): filename for the pickle file
    
        Returns: 
            None
         """  
        if self.data is None:
            print("There's no data to save. Run DoeVars.sample_doevars() first")
        else:
            self.data.to_pickle(filename)

if __name__ == '__main__':






     ### COMBINATIONS and sampling
    vars = {'Fs': SalibSobol(4, {'F11':[10, 20], 'F12':[-0.1,0.15], 'F22':[-0.15, 1]}), 
            'R': [0.3, 0.5], 
            'particle': ({'P': [100, 200]}, {'Q': [15, 25]} ) 
            }


    f = SalibSobol(2, {'F11':[10, 20], 'F12':[-0.1,0.15], 'F22':[-0.15, 1]})

    fs = f.compute_sampling()
    


    # print(fs)
   
    fd = {'F11':[10, 20], 'F12':[-0.1,0.15], 'F22':[-0.15, 1]}
    
    # print(fs)
    # print(list(fd.keys()))
    #for Arrays, samples
    #convert to data framed from array and columns
    df1 = pd.DataFrame(data=fs, columns=list(fd.keys()))
    # print(df1 )

    # combinations 
    r = {'R': [0.3, 0.5]}
    df2 = pd.DataFrame.from_dict(r)
    # print(df2)

    # using merge: many-to-many, cartesian product
    merge_df =df1.merge(df2, how='cross')
    # print(merge_df)

    # many-to-one using index on df1
    # merge_df2 = pd.merge(merge_df, df2, how='left')
    p = {'particle.P': [100, 200]}
    q = {'particle.Q': [15, 25 ]}
    df3 = pd.DataFrame.from_dict(p)
    df4 = pd.DataFrame.from_dict(q)

    merged_df2 = df3.append(df4) 
    # print(merged_df2)

    # many to many for cartesian product
    merged_df3 = merge_df.merge(merged_df2, how='cross')

    # print(merged_df3)

    # doe = DoeVars(vars)

    # print('DoEVars definition:')
    # print(doe)

    # print('\n DoEVars summary information:')
    # print(doe.info())

    # Compute sampling and combinations
    # doe.do_sampling()