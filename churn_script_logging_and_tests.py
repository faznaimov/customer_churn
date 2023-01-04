'''
Module to test churn_library.py

Author: Faz Naimov
Date: 1/3/2023
'''

import os
import logging
import churn_library as cls

logging.basicConfig(
	filename='./logs/churn_library.log',
	level=logging.INFO,
	filemode='w',
	format='%(name)s - %(levelname)s - %(message)s')


def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = cls.import_data('data/bank_data.csv')
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error(
			"Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	'''
	test perform eda function
	'''
	df = cls.import_data("data/bank_data.csv")
	# call the function under test
	cls.perform_eda(df)

	# set expectations for output files
	file_names = [
		'./images/churn.png',
		'./images/age_dist.png',
		'./images/marital_status.png',
		'./images/total_trans_ct.png',
		'./images/corr.png'
	]
	try:
		# check if expected files exist
		assert all(os.path.isfile(file_name) for file_name in file_names)

		# check if expected files are not empty
		assert all(os.stat(file_name).st_size >
				   0 for file_name in file_names)

		logging.info("Testing test_eda: SUCCESS")
	except AssertionError as err:
		logging.error("Testing test_eda: Some files were not found or empty %s",
					  list(filter(lambda x: not os.path.isfile(x), file_names)))
		raise err


def test_encoder_helper():
	'''
	test encoder helper
	'''
	pass

def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''
	pass

def test_train_models():
	'''
	test train_models
	'''
	pass

if __name__ == "__main__":
	# test all functions and report final result
	test_import()
	test_eda()