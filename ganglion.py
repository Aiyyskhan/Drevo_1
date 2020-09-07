"""
**************************************

source code: PythonNeuralNetwork/keras_examples/ganglion_20200113/ganglion_v.0.1.py
date: 01.05.2020

**************************************
"""

import numpy as np
# import tensorflow as tf


class Ganglion:
	def __init__(self, num_inputs, num_outputs, num_units=10, connections_limit=1000):
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.num_units = num_units
		self.connections_limit = connections_limit

		self.index_list = []
		self.synapse_type_list = []
		self.synapse_weight_list = []
		
		self.potentials = np.zeros(self.num_units, dtype=np.float32)

		self._build_indices()
		self._build_connections()
		self._build_weights()

	def __call__(self, input_data): #, w_mode=True):
		# матрица весов или матрица типов
		input_data = np.array(input_data)
		# syn_list = self.synapse_weight_list if w_mode else self.synapse_type_list
		
		self.potentials[self.index_list[0][1]] += np.dot(input_data[self.index_list[0][0]], (self.synapse_type_list[0] * self.synapse_weight_list[0]))
		
		for idx, syn_matrix in enumerate(self.synapse_type_list[1:-1]):
			self.potentials[self.index_list[idx+1][1]] += np.dot(self.__activation_function(self.potentials[self.index_list[idx+1][0]]), (syn_matrix * self.synapse_weight_list[idx+1]))
		
		out = self.__activation_function(np.dot(self.__activation_function(self.potentials[self.index_list[-1][0]]), (self.synapse_type_list[-1] * self.synapse_weight_list[-1])))
		
		# self.potentials *= 0.0
		self.__potential_decrement()
		self.__potential_limiter()

		return out

	def add_units(self):
		if len(self.index_list[-2][0]) < self.connections_limit:
			if self.num_units > self.connections_limit:
				self.index_list[-2][0] = np.concatenate([self.index_list[-2][0], [self.num_units]])
				self.index_list[-3][1] = np.concatenate([self.index_list[-3][1], [self.num_units]])

				self.synapse_type_list[-2] = self.__add_connections(self.synapse_type_list[-2], 0)
				self.synapse_type_list[-3] = self.__add_connections(self.synapse_type_list[-3], 1)

			elif self.num_units < self.connections_limit:
				self.index_list[0][1] = np.concatenate([self.index_list[0][1], [self.num_units]])
				self.index_list[-2][1] = np.concatenate([self.index_list[-2][1], [self.num_units]])
				self.index_list[-2][0] = np.concatenate([self.index_list[-2][0], [self.num_units]])
				self.index_list[-1][0] = np.concatenate([self.index_list[-1][0], [self.num_units]])

				self.synapse_type_list[0] = self.__add_connections(self.synapse_type_list[0], 1)
				self.synapse_type_list[-2] = self.__add_connections(self.synapse_type_list[-2], 1)
				self.synapse_type_list[-2] = self.__add_connections(self.synapse_type_list[-2], 0)
				self.synapse_type_list[-1] = self.__add_connections(self.synapse_type_list[-1], 0)

		else:
			end_segment_indices = self.index_list.pop()
			new_index = np.array([self.num_units])

			type_end_segment = self.synapse_type_list.pop()

			row_indices = np.random.permutation(np.arange(self.num_units))[:self.connections_limit]
			self.index_list.append([row_indices, new_index])

			column_indices = np.random.permutation(np.arange(self.num_units))[:self.connections_limit]
			self.index_list.append([new_index, column_indices])

			self.index_list.append(end_segment_indices)

			syn_type = self.__builder(self.connections_limit, 1)
			self.synapse_type_list.append(syn_type)

			syn_type = self.__builder(1, self.connections_limit)
			self.synapse_type_list.append(syn_type)

			self.synapse_type_list.append(type_end_segment)

		self._build_weights()
		self.num_units += 1
		self.potentials = np.zeros(self.num_units)

	def get_parameters(self):
		return [self.num_inputs, self.num_outputs, self.num_units, self.connections_limit], self.index_list

	def get_synapse_weights(self):
		return self.synapse_weight_list

	def get_synapse_types(self):
		return self.synapse_type_list

	def _build_indices(self):
		# если кол-во нейронов больше лимита, то
		if self.num_units > self.connections_limit:
			# создаются векторы индексов строк и столбцов
			row_indices = np.arange(self.num_inputs)
			# индексы столбцов перемешаны для случайного порядка и ограничены лимитом
			column_indices = np.random.permutation(np.arange(self.num_units))[:self.connections_limit]
			# индексы добавляем в список индексов
			self.index_list.append([row_indices, column_indices])

			# создается временный вектор перемешанных в случайном порядке индексов нейронов
			unit_indices = np.random.permutation(np.arange(self.num_units))
			# пока длина этого вектора больше нуля
			while len(unit_indices) > 0:
				# если длина вектора больше лимита, то
				if len(unit_indices) > self.connections_limit:
					# создаются векторы индексов строк и столбцов
					row_indices = unit_indices[:self.connections_limit]
					# индексы столбцов перемешаны для случайного порядка и ограничены лимитом
					column_indices = np.random.permutation(np.arange(self.num_units))[:self.connections_limit]
					# индексы добавляем в список индексов
					self.index_list.append([row_indices, column_indices])
					
					# укорачиваем временный индекс
					unit_indices = unit_indices[self.connections_limit:]
				# иначе
				else:
					# создаются вектор индексов строк, ограниченный лимитом
					row_indices = np.random.permutation(np.arange(self.num_units))[:self.connections_limit]
					# вектору индексов столбцов присваиваются значения временного вектора
					column_indices = unit_indices
					# индексы добавляем в список индексов
					self.index_list.append([row_indices, column_indices])

					# вектору индексов строк присваиваются значения временного вектора
					row_indices = unit_indices
					# создается вектор индексов столбцов, ограниченный лимитом
					column_indices = np.random.permutation(np.arange(self.num_units))[:self.connections_limit]
					# индексы добавляем в список индексов
					self.index_list.append([row_indices, column_indices])
					
					# укорачиваем временный индекс до нуля
					unit_indices = np.arange(0)

			row_indices = np.random.permutation(np.arange(self.num_units))[:self.connections_limit]
			column_indices = np.arange(self.num_outputs)
			self.index_list.append([row_indices, column_indices])

		else:
			row_indices = np.arange(self.num_inputs)
			column_indices = np.random.permutation(np.arange(self.num_units))
			self.index_list.append([row_indices, column_indices])

			row_indices = np.random.permutation(np.arange(self.num_units))
			column_indices = np.random.permutation(np.arange(self.num_units))
			self.index_list.append([row_indices, column_indices])

			row_indices = np.random.permutation(np.arange(self.num_units))
			column_indices = np.arange(self.num_outputs)
			self.index_list.append([row_indices, column_indices])

	def _build_connections(self):
		for idx in self.index_list:
			syn_type = self.__builder(len(idx[0]), len(idx[1]))
			self.synapse_type_list.append(syn_type)

	def _build_weights(self):
		self.synapse_weight_list.clear()
		for syn_type_matrix in self.synapse_type_list:
			syn_weight_init = np.random.uniform(0.01, 5.0, (syn_type_matrix.shape)).astype(np.float32)
			self.synapse_weight_list.append(syn_weight_init)

	def __builder(self, rows: int, columns: int):
		return np.random.randint(-1, 2, size=(rows, columns)).astype(np.float32)
	
	def __add_connections(self, type_array, axis):
		shape = type_array.shape
		if axis==0:
			t_v_segment = self.__builder(1, shape[1])
			type_array = np.vstack([type_array, t_v_segment])
		else:
			t_h_segment = self.__builder(shape[0], 1)
			type_array = np.hstack([type_array, t_h_segment])
		return type_array

	def __activation_function(self, s_vector):
		return 1 / (1 + np.exp(-s_vector)) #-10.0))
	
	# def __activation_function(self, s_vector):
	# 	out_vector = s_vector.copy()
	# 	out_vector[out_vector <= 0.0] = 0.0
	# 	out_vector = out_vector / 5.0
	# 	out_vector[out_vector > 1.0] = 1.0
	# 	return out_vector

	# def __activation_function(self, s_vector):
	# 	return np.vectorize(self.custom_clippedReLU)(s_vector)

	# def custom_clippedReLU(self, s):
	# 	if s <= 0.0:
	# 		return 0.0
	# 	elif s > 10.0:
	# 		return 1.0
	# 	else:
	# 		return s / 10.0


	def __potential_decrement(self):
		positive_condition = self.potentials > 0.001
		negative_condition = self.potentials < -0.001
		near_zero_condition = np.abs(self.potentials) <= 0.001

		if np.any(positive_condition):
			self.potentials[positive_condition] -= self.potentials[positive_condition] * 0.75
		if np.any(negative_condition):
			self.potentials[negative_condition] += self.potentials[negative_condition] * 0.75
		if np.any(near_zero_condition):
			self.potentials[near_zero_condition] *= 0

	def __potential_limiter(self):
		self.potentials[self.potentials > 20.0] = 20.0
		self.potentials[self.potentials < -5.0] = -5.0