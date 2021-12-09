
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


def find_sampling_vars(doe_vars: dict):
    """
    Find names of DoeVars that contain a definiton of a sampling method.
    WARNING: only 3 levels of nesting are checked in the input.
    Args:
        doe_vars (dict): variables defining a design of experiment
    Returns:
        list of names
    """
    # expand dictionary
    df = pd.json_normalize(doe_vars)
    vars_level1 = df.to_dict(orient='records')[0]

    elements_with_functions = [] # list of names
    # loop over variables at the top level of serialization
    for key, value in vars_level1.items():
        # id to keep track of which variable belog to the same group
        # group ids will will start with f3dams# followed by the value group_id
        group_id = 0
        if isinstance(value, SamplingMethod):
            elements_with_functions.append(key)

        # case for OR operator, 
        # variable grouped as a tupple are combined differently
        # then ungrouped variables, see create_combinations()
        elif isinstance(value, tuple):
            for grouped_var in value:
                g_df = pd.json_normalize(grouped_var)
                g_vars = g_df.to_dict(orient='records')[0]
                for g_key, g_value in g_vars.items():
                    if isinstance(g_value, SamplingMethod):
                        elements_with_functions.append('f3dasm#'+str(group_id)+'.'+key+'.'+g_key)
        else:
            continue
        group_id+=1
    # TODO: find a generic solution this function. So that it works for 
    # any level of nesting.
    return elements_with_functions


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
        
        print('sampling var:', self.sampling_vars)

        # Compute sampling for variable with a sampling method
        for var in self.sampling_vars:
            inner_vars = var.split('.') 
            # print('inner vars:', inner_vars)
            # CASES:
            ## sampling at the top level 
            if len(inner_vars) == 1:

                df1 =apply_sampling(doe_vars[var], list(doe_vars[var].sampling_ranges.keys()))

                print(df1)

            ## sampling variable at a second level that DO NOT define a group 
            # (use OR operator). Groups contains an element which starts with 
            # the string 'f3dasm#'
            elif not "f3dasm#" in inner_vars[0]: 
                print(inner_vars)
                if len(inner_vars) == 2:
                    df2 = apply_sampling(doe_vars[inner_vars[0]][inner_vars[1]], [ inner_vars[0]+'.'+inner_vars[1] ])

                    print('df2', df2)

                elif len(inner_vars) == 3:
                    
                    df3 = apply_sampling(
                                    doe_vars[inner_vars[0]][inner_vars[1]][inner_vars[2]], 
                                    [ inner_vars[0]+'.'+inner_vars[1]+'.'+inner_vars[2]]
                                    )                    
                    print(df3)
                else:
                    raise ValueError("DoeVars definition contains too many nested elements. A max of 3 is allowed")

            ## sampling for varibles in a group (OR operator)
            elif "f3dasm#" in inner_vars[0]:
                 
                # print(inner_vars)
                # remove group id
                inner_v = copy.deepcopy(inner_vars)
                inner_v.pop(0)
                print('inner',inner_v)

                # items in the tupple
                t_items = doe_vars[inner_v[0]]
                counter = 0

                for item in t_items:
                    print(item)
                # while counter < no_items:
                #     print(inner_v[0], counter, inner_v[1], inner_v[2])
                #     try:
                #         print(doe_vars[ inner_v[0] ] [counter] [inner_v[1]] )
                #     except:
                #         print('skip')
                #     else:
                #         if len(inner_vars) == 1:

                #             df5 =apply_sampling(doe_vars[var], list(doe_vars[var].sampling_ranges.keys()))

                #             print(df5)
                #         if len(inner_v) == 2:
                #             df4 = apply_sampling(doe_vars[inner_v[0]][counter], [ inner_v[0]+'.'+inner_v[1] ])

                #             print('df4', df4)

                #         elif len(inner_v) == 3:
                #             print(3)
                    
                #         #     df3 = apply_sampling(
                #         #             doe_vars[inner_v[0]][counter][inner_v[2]], 
                #         #             [ inner_v[0]+'.'+inner_v[1]+'.'+inner_v[2]]
                #         #             )                    
                #         #     print(df3)
                #         else:
                #             raise ValueError("DoeVars definition contains too many nested elements. A max of 3 is allowed")

                #     counter+=1
                    
                
                # for item in range(len(doe_vars[inner_v[0]])): # iterate thru tupple
                #     print(item)
                #     print(inner_v[0], item, inner_v[1], inner_v[2])
                #     print(doe_vars[ inner_v[0] ] [0] [inner_v[1]] )
                    # print('item2', item['NeoHookean'])
                #     print(item[inner_vars[1]])

                                

                # # a group must contain at least 2 elements besided the group id
                # if len(inner_vars) == 2: # check if group (tuple) contains only one element
                #     raise SyntaxError(f'variable {inner_vars} defines a group with a single parameter. Remove ()')
                
                # if len(inner_vars) == 3: # check group has 2 elements
                #     # remove group id
                #     inner_vars.pop(0)
                    
                # elif len(inner_vars) == 4: # check group has 3 elements
                #     pass
                # else:
                #     raise ValueError("Group definition contains too many nested elements. A max of 3 is allowed")


            else:
                pass
            
            # sampling variables at second that belong to a group
    



            #     print("var-ranges:", doe_vars[var].sampling_ranges.keys())
            #     doe_vars[var] = 1# samples_to_dict( doe_vars[var].compute_sampling(), doe_vars[var].sampling_ranges.keys())
            #     print('as dict', doe_vars[var])
            # elif len(inner_vars) == 2:
            #     doe_vars[inner_vars[0]][inner_vars[1]] = doe_vars[inner_vars[0]][inner_vars[1]].compute_sampling()
            # elif len(inner_vars) == 3:
            #     doe_vars[inner_vars[0]][inner_vars[1]][inner_vars[2]] = doe_vars[inner_vars[0]][inner_vars[1]][inner_vars[2]].compute_sampling()
            # else:
            #     raise SyntaxError("DoeVars definition contains too many nested elements. A max of 3 is allowed")
    
        # print('sampling doe', doe_vars)
        # combinations
        # print(doe_vars)

        # sampled_values = list( deserialize_dictionary(doe_vars).values() )
        # combinations = create_combinations(numpy.meshgrid, sampled_values)

        # # dataframe
        # _columns =list( deserialize_dictionary(doe_vars).keys() )
        # self.data = pd.DataFrame(combinations,columns=_columns)
        # return self.data

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