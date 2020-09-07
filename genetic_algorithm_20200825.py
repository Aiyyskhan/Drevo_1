"""
**************************************

for ganglion_base_nn_20200115.py
date: 25.08.2020

**************************************
"""

import numpy as np
import random

def selection(population: list, results: list, num_leader: int, sort_by_max: bool = False) -> list:
	"""функция отбора"""

	# сортировка по результатам
	if sort_by_max: 
		indices = np.argsort(np.array(results))[-1:-(num_leader+1):-1]
	else: 
		indices = np.argsort(np.array(results))[:num_leader]
	
	# отбор лидеров
	return [population[idx] for idx in indices].copy()

def crossover(leader_arr: list, population_size: int) -> list:
	"""функция кроссинговера"""

	indiv_arr = [leader_arr[0].copy() for _ in range(population_size)]

	max_percents = np.linspace(11, 100, len(leader_arr)).astype(int)[::-1]
	parents_prob_list = random.choices(list(range(len(leader_arr))), list(range(len(leader_arr))[::-1]), k=population_size)

	for i, individ in enumerate(indiv_arr[1:]):
		leader_1_index = parents_prob_list[i]
		leader_2_index = np.random.permutation(parents_prob_list)[i]
		while leader_1_index == leader_2_index:
			leader_2_index = np.random.permutation(parents_prob_list)[i]
		for j in range(len(individ)):
			individ[j] = __hybridization(leader_arr[leader_2_index][j].copy(), leader_arr[leader_1_index][j].copy(), max_percents[leader_1_index])

	return indiv_arr

def mutation(population: list, syn_weight=True) -> list:
	"""функция мутации"""

	max_percents = np.linspace(10, 50, len(population)-1).astype(int)
	
	for idx, individ in enumerate(population[1:]):
		for i in range(len(individ)):
			individ[i] = __mutate(individ[i].copy(), syn_weight, max_percents[idx])

	return population

def __hybridization(lead_2_arr: np.ndarray, lead_1_arr: np.ndarray, max_percent: int) -> np.ndarray:

	percent = np.random.randint(10, max_percent)

	arr_form = lead_2_arr.shape
	ind_arr_rav = lead_2_arr.ravel()
	lead_arr_rav = lead_1_arr.ravel()

	len_ind_arr = len(ind_arr_rav)
	mixed_indices = np.arange(len_ind_arr)
	np.random.shuffle(mixed_indices)

	num_mixed_indices = np.round((percent / 100) * len_ind_arr).astype('int')
	mixed_index_slice = mixed_indices[:num_mixed_indices]

	ind_arr_rav[mixed_index_slice] = lead_arr_rav[mixed_index_slice]
	
	return ind_arr_rav.reshape(arr_form)

def __mutate(ind_arr: np.ndarray, syn_weight, max_percent: int) -> np.ndarray:
	
	percent = np.random.randint(0, max_percent)

	arr_form = ind_arr.shape
	ind_arr_rav = ind_arr.ravel()

	len_ind_arr = len(ind_arr_rav)
	mixed_indices = np.arange(len_ind_arr)
	np.random.shuffle(mixed_indices)
	
	num_mixed_indices = np.round((percent / 100) * len_ind_arr).astype('int')
	mixed_index_slice = mixed_indices[:num_mixed_indices]

	if syn_weight:
		ind_arr_rav[mixed_index_slice] = np.abs(ind_arr_rav[mixed_index_slice] + np.random.uniform(-1.0, 1.0, num_mixed_indices).astype(np.float32)) 
	else:
		ind_arr_rav[mixed_index_slice] = np.random.randint(-1, 2, num_mixed_indices)

	return ind_arr_rav.reshape(arr_form)