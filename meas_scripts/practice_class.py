# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:23:37 2025

@author: joaop
"""

import pickle

class example_class:
    a_number = 35
    a_string = "hey"
    a_list = [1, 2, 3]
    a_dict = {"first":"a", "second": 2, "third": [1,2,3]}
    a_tuple = (22,23)
    
    
my_object = example_class() # this is a instance of my class

my_pickled_object = pickle.dumps(my_object) # creating the pickle object
                                            # saving in string format 
                                            # usando pickle.DUMPS

print(f"This is my pickled object: {my_pickled_object}")

#%%

my_object.a_dict = None 

my_unpickled_object = pickle.loads(my_pickled_object)   # creating the pickle object
                                                        # saving in string format 
                                                        # usando pickle.LOADS

print(f"a_dict of unpickled object: {my_unpickled_object.a_dict}")

#%%
