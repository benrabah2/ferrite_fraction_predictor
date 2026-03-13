#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:27:25 2024

@author: benrabah2
"""

## data utils

import numpy as np
from scipy.interpolate import interp1d
from tensorflow.keras.utils import normalize

def Convert_2theta_to_q(X_dict_data, wave_length, q_min, q_max):
    X_dict_data[:, 0] = np.abs(np.sin(np.radians((X_dict_data[:, 0]) / 2)) * 4 * np.pi / wave_length)
    index_q = np.where(np.logical_and(X_dict_data[:, 0] >= q_min, X_dict_data[:, 0] <= q_max))[0]
    X_dict_data = X_dict_data[index_q, :]
    return X_dict_data

def Shape_Correction_Function(X_dict_data, nb_of_points_per_profile):
    original_array = X_dict_data[:, 1]
    original_array_x = X_dict_data[:, 0]
    index_shape = np.linspace(np.min(original_array_x), np.max(original_array_x), len(original_array))
    interpolation_func = interp1d(index_shape, original_array, kind='linear')
    new_index_shape = np.linspace(np.min(original_array_x), np.max(original_array_x), nb_of_points_per_profile)
    interpolated_array = interpolation_func(new_index_shape)
    X_dict_data = np.column_stack((new_index_shape, interpolated_array))
    return X_dict_data

def normalize_array(arr, target_max):
    '''
    This function normalizes the array to the target_max
    '''
    max_value = np.max(arr)
    normalized_arr = arr / max_value
    normalized_arr *= target_max
    return normalized_arr

def Normalize(data_array):
    
    '''
    This function normalize the X-array using TF normalization function
    '''
    
    normalized_array = np.empty_like(data_array)

    for i in range(data_array.shape[2]):
        column = data_array[:, :, i]                                             # Extract the column
        normalized_column = normalize(column, axis=1)                           # Normalize the column along axis=1
        normalized_array[:, :, i] = normalized_column                           # Assign the normalized column to the new array

    return normalized_array


