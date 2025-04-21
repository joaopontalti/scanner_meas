# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:41:08 2024

#########################################################

**Owner:** Joao Vitor Silva Pontalti - Undergraduate student  
**Advisor:** Eric Brandão Carneiro - Prof. Dr. Eng.
---
## 	**ACOUSTICAL ENGINEERING**  
**UNIVERSIDADE FEDERAL DE SANTA MARIA**

**Last update:** 16/01/2025
---
# Post Processing used for my TCC
@author: joaop
"""
#%% Importing stuff
import numpy as np # Primordial one
import matplotlib.pyplot as plt # For plotting
import pytta # For object manipulation
 
from controlsair import AirProperties, AlgControls # for atmospheric control
from receivers import Receiver # for arrays of microphones creation
from sources import Source # for source properties 
from qterm_estimation import ImpedanceDeductionQterm # For *GOING PRO* DSP
from ppro_meas_insitu import InsituMeasurementPostPro # Creates post pro object

from mpl_toolkits.mplot3d.art3d import Poly3DCollection #for scenary plots
from mpl_toolkits.mplot3d import Axes3D

#%% Import your files 

# Path to the folder where you saved your measurements
main_folder = 'C:/Users/joaop/anaconda3/envs/Dataset_JoaoP/TCC_DATASET_JoaoP/DATA2025/'

# name of your measurement (as saved in your measurement object) 
name = 'PU_L48cm_d2cm_s100cm_2mics_15012025_1' 

#%% Post Pro object creatin - it will load the meas_obj
ppro_obj = InsituMeasurementPostPro(main_folder = main_folder, name = name, t_bypass=0)


#%% Compute all IR
ppro_obj.compute_all_ir_load(regularization = True,  deconv_with_rec = True, 
                   only_linear_part = True)

#%% Load all IR 
ppro_obj.load_irs()

#%%