# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:56:53 2025

@author: joaop

.py file containing a bunch of functions I created to auxiliate programming

"""

# Import the important things

import matplotlib.pyplot as plt
import numpy as np
import pytta
import os 
import re

from controlsair import AirProperties, AlgControls#, add_noise, add_noise2
from sources import Source
from receivers import Receiver

from ppro_meas_insitu import InsituMeasurementPostPro

from decomposition_ev_ig import DecompositionEv2, ZsArrayEvIg, filter_evan
from qterm_estimation import ImpedanceDeductionQterm
from scipy.interpolate import griddata

from pathlib import Path
from matplotlib.ticker import FuncFormatter

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
    
from decomp_quad_v2 import Decomposition_QDT
from decomp2mono import Decomposition_2M  # Monopoles

########################################################################################
# --
########################################################################################


def convert_to_meters(value):
    """
    Function to convert values given in cm ou mm to m. It does that
    by searching the sulfix and operating multiplication accordingly.
    """
    if "cm" in value:
        # Remove "cm" e converte para metros (1 cm = 0.01 metros)
        return float(value.replace("cm", "")) / 100
    elif "mm" in value:
        # Remove "mm" e converte para metros (1 mm = 0.001 metros)
        return float(value.replace("mm", "")) / 1000
    else:
        # Caso o valor já esteja em metros ou não tenha unidade, retorna o valor original
        return float(value)

def select_mics(ppro_obj, index, baffle_size, par_index, elev=0, azim=0, output_folder=".",plot=False):
    '''
    # Selecting a Microphone Position and Finding its Relative Pair

    This function helps you select a microphone position and find its relative microphone, 
    which is positioned either above or below the one you initially chose.

    Parameters:
    - **Coords**: 
      A matrix of size N x 3 containing the spatial coordinates of the microphone positions.
    - **index**: 
      The index of the measurement you want to examine. It must be an integer within the range 
      of available measurements. For example, in an array with 128 measurement points, you can 
      choose an index from 0 to 127.

    ## Usage Recommendation:
    1. Insert an `index` (e.g., 42) to find its relative pair.
    2. Verify the result by using the relative index obtained as the new input. 
       This ensures the function correctly identifies matching microphone pairs.

    This step-by-step process helps confirm the consistency of the pairing logic.
    '''
    coords = ppro_obj.meas_obj.receivers.coord
    coord_index = coords[index]
    pares = []
    par_index=None
    
    for i, coord in enumerate(coords): # for all the enumerated coord lines
        if i != index:  # the index must be different (just a way to avoid redundancy)
            if coord[0] == coord_index[0] and coord[1] == coord_index[1] and coord[2] != coord_index[2]:
                pares.append((i, coord))  # save the index and the coordinate that 
                                          # suits the conditions 
    if pares:  # Check if there are any valid pairs
        par_index = pares[0][0]  # Save the first matching pair index (if any)
        
    title_name, _, _, _, _, Lx_value, Ly_value, Lz_value = format_meas_name(ppro_obj.meas_obj.name, basic_infos=False)

    L_x = Lx_value 
    L_y = Ly_value
    sample_thickness = Lz_value
    baffle_size = 1.2
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        list_of_sample_verts = [
            np.array([[-L_x/2, -L_y/2, 0.0], [L_x/2, -L_y/2, 0.0], [L_x/2, L_y/2, 0.0], [-L_x/2, L_y/2, 0.0]]),
            np.array([[-L_x/2, -L_y/2, 0.0], [L_x/2, -L_y/2, 0.0], [L_x/2, -L_y/2, -sample_thickness], [-L_x/2, -L_y/2, -sample_thickness]]),
            np.array([[-L_x/2, L_y/2, 0.0], [-L_x/2, L_y/2, -sample_thickness], [L_x/2, L_y/2, -sample_thickness], [L_x/2, L_y/2, 0.0]]),
            np.array([[-L_x/2, -L_y/2, 0.0], [-L_x/2, -L_y/2, -sample_thickness], [-L_x/2, L_y/2, -sample_thickness], [-L_x/2, L_y/2, 0.0]]),
            np.array([[L_x/2, -L_y/2, 0.0], [L_x/2, L_y/2, 0.0], [L_x/2, L_y/2, -sample_thickness], [L_x/2, -L_y/2, -sample_thickness]])
        ]
        
        for verts in list_of_sample_verts:
            collection = Poly3DCollection([list(zip(verts[:, 0], verts[:, 1], verts[:, 2]))],
                                          linewidths=0.5, alpha=0.5, edgecolor='tab:blue', zorder=2)
            collection.set_facecolor('tab:blue')
            ax.add_collection3d(collection)
        
        baffle = np.array([[-baffle_size/2, -baffle_size/2, -sample_thickness],
                            [baffle_size/2, -baffle_size/2, -sample_thickness],
                            [baffle_size/2, baffle_size/2, -sample_thickness],
                            [-baffle_size/2, baffle_size/2, -sample_thickness]])
        
        collection = Poly3DCollection([list(zip(baffle[:, 0], baffle[:, 1], baffle[:, 2]))],
                                      linewidths=0.5, alpha=0.5, edgecolor='grey', zorder=2)
        collection.set_facecolor('grey')
        ax.add_collection3d(collection)
        
        # Plot source
        if ppro_obj.meas_obj.source is not None:
            ax.scatter(ppro_obj.meas_obj.source.coord[0, 0],
                       ppro_obj.meas_obj.source.coord[0, 1],
                       ppro_obj.meas_obj.source.coord[0, 2],
                       s=200, marker='*', color='red', alpha=0.5)
    
        # Plot receivers  # we plot each point of the r_coord [[x],[y],[z]]
        for r_coord in range(ppro_obj.meas_obj.receivers.coord.shape[0]):
            ax.scatter([ppro_obj.meas_obj.receivers.coord[r_coord, 0]],
                       [ppro_obj.meas_obj.receivers.coord[r_coord, 1]],
                       [ppro_obj.meas_obj.receivers.coord[r_coord, 2]],
                       marker='o', s=12, color='blue', alpha=0.7)
        
        
        # ADDED: Plot selected pair position 
        ax.scatter(coord_index[0], coord_index[1], coord_index[2], 
                   s = 50, color='red', marker = 'o', label='Index')
        for i, (par_index, par) in enumerate(pares, start=1):
            ax.scatter(par[0], par[1], par[2], s=50, color='green', label=f'Par {i}' if i == 1 else "")
    
        ax.set_title(f"Measurement scene - 2 mics selected {index} e {par_index}\n {title_name}\n")
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.set_zlabel(r'$z$ [m]')
        ax.grid(False)
        ax.set_xlim((-baffle_size/2, baffle_size/2))
        ax.set_ylim((-baffle_size/2, baffle_size/2))
        ax.set_zlim((-0.1, 1.0))
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        plt.show()
    
    return par_index, title_name

########################################################################################
# --
########################################################################################

    
def sph2cart(r, theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters:
    r (float): Radius
    theta (float): Inclination (elevation) in radians
    phi (float): Azimuth in radians

    Returns:
    tuple: (x, y, z) Cartesian coordinates
    """
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z

########################################################################################
# --
########################################################################################

def list_folders(main_direct, keyword):
    if not os.path.exists(main_direct):
        print(f"Invalid pathway. Check again, dude: {main_direct}")
        return
    
    # im using a list comprehension to name the desired folders
    # read as: for all folders contained in main_direct, create a list with
    # all the ones that contained a specific keyword 
  
    folders = [f for f in os.listdir(main_direct) if keyword in f]

    
    if folders:
        print("Folders found:")
        for i, folder in enumerate(folders):
            print(f"{i}: {folder}")
    else:
        print("No folders were found with the given keyword.")
        
    select_index = input("\n Select folders to process (separate the numbers with commas): ")
    select_index = [int(i.strip()) for i in select_index.split(",")
                    if i.strip().isdigit()]
    
    select_folders = [folders[i] for i in select_index if 0 <= i < len(folders)]
    
    return select_folders

#def meas_info(meas_name, basic_infos=False):
    """
    Função para extrair informações do nome da medição de forma clara.

    Parameters:
    meas_name (str): Nome da medição, como por exemplo 'AbsTriangOrange_L60cm_d5cm_s100cm_2planar_14012025'.
    basic_infos (bool): Se True, retorna apenas informações básicas, se False, retorna todas as informações detalhadas.

    Returns:
    dict: Dicionário com as informações extraídas.
    """
    # Inicializando o dicionário de resultados
    extracted_info = {}
    
    # Dividir o nome em partes
    parts = meas_name.split('_')
    
    # Verificando se o nome tem o formato esperado e extraindo informações
    if len(parts) >= 5:
        extracted_info['Material'] = parts[0]  # Ex: AbsTriangOrange
        extracted_info['Lx'] = parts[1]  # Ex: L60cm
        extracted_info['d'] = parts[2]  # Ex: d5cm
        extracted_info['s'] = parts[3]  # Ex: s100cm
        extracted_info['Config'] = parts[4]  # Ex: 2planar
        
        # Data no final do nome
        extracted_info['Date'] = parts[-1]  # Ex: 14012025
        
       # Se basic_infos for True, imprime apenas as informações básicas
        if basic_infos:
            print(f"Material: {extracted_info['Material']}")
            print(f"Lx: {extracted_info['Lx']}")
            print(f"Ly: {parts[5] if len(parts) > 5 else 'N/A'}")
            print(f"Lz: {parts[6] if len(parts) > 6 else 'N/A'}")
        else:
            # Caso precise de informações detalhadas
            extracted_info['Details'] = {
                'Ly': parts[5] if len(parts) > 5 else 'N/A',  # Ex: Ly36cm (opcional)
                'Lz': parts[6] if len(parts) > 6 else 'N/A',  # Ex: Lz10cm (opcional)
            }
    
    else:
        # Caso o formato não seja o esperado, retornamos um erro mais amigável
        raise ValueError(f"Nome de medição inválido ou formato não reconhecido: {meas_name}")
    
    return extracted_info


import re

def meas_info(meas_name, basic_infos=False):
    """
    Extrai os valores de Lx, Ly e Lz (ou d como Lz) do nome da medição.

    Parameters:
    meas_name (str): Nome da medição, ex: 'PETWool_Lx60cm_Ly36cm_Lz10cm_2mics_28012025_3'.
    basic_infos (bool): Se True, imprime os valores extraídos.

    Returns:
    tuple: (Lx, Ly, Lz) em centímetros, ou 'N/A' se não encontrado.
    """
    # Extração dos valores
    Lx = re.search(r'Lx(\d+)cm', meas_name)
    Ly = re.search(r'Ly(\d+)cm', meas_name)
    Lz = re.search(r'(Lz|d)(\d+)cm', meas_name)  # Lz ou d como Lz
    
    # Verificando se as correspondências foram encontradas
    Lx_val = int(Lx.group(1)) if Lx else 'N/A'
    Ly_val = int(Ly.group(1)) if Ly else 'N/A'
    Lz_val = int(Lz.group(2)) if Lz else 'N/A'

    # Caso não tenha encontrado Lz e tenha "d", vamos considerar "d" como Lz
    if Lz_val == 'N/A' and 'd' in meas_name:
        Lz_val = int(re.search(r'd(\d+)cm', meas_name).group(1)) if re.search(r'd(\d+)cm', meas_name) else 'N/A'

    if basic_infos:
        print(f"Lx: {Lx_val} cm")
        print(f"Ly: {Ly_val} cm")
        print(f"Lz: {Lz_val} cm")

    return Lx_val, Ly_val, Lz_val


######## TESTAR

def plot_ir_with_annotations(ppro, rec_index=1, xlims=(0, 25e-3)):
    """
    Plota a Resposta Impulsiva (IR) com formatação personalizada e anotações.

    Parâmetros:
    - ppro: Objeto contendo os dados da resposta impulsiva.
    - rec_index: Índice do receptor a ser plotado.
    - xlims: Limites do eixo X em segundos.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot da IR normalizada
    plt.plot(ppro.time_ht * 1000, ppro.ht_mtx[rec_index, :] / np.amax(ppro.ht_mtx[rec_index, :]), label='IR')
    
    # Ajuste de ticks do eixo X e Y
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Definição dos formatadores personalizados
    def format_ticks_y(x, pos):
        return f'{x:.1f}'.replace('.', ',')
    
    def format_ticks_x(x, pos):
        return f'{x:.3f}'.replace('.', ',')
    
    # Aplicar formatadores
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks_x))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks_y))
    
    # Definir limites do eixo X
    plt.xlim(xlims)
    
    # Anotações
    plt.annotate('Primeiras reflexões\nda sala',
                 xy=(15e-3, 0.02),  # Coordenada do ponto
                 xytext=(15e-3, 0.23),  # Posição do texto
                 textcoords='data',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightyellow'),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=2),
                 fontsize=25)
    
    plt.annotate('Região de interesse',
                 xy=(5e-3, 5e-3),
                 xytext=(2.5e-3, 0.575),
                 textcoords='data',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightyellow'),
                 fontsize=25)
    
    plt.tight_layout()
    plt.show()

