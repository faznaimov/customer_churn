'''
Module to test churn_library.py

Author: Faz Naimov
Date: 1/3/2023
'''

import os
import logging
import pandas as pd
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data, pth):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        test_df = import_data(pth)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: FAILED The file wasn't found")
        raise err

    try:
        assert test_df.shape[0] > 0
        assert test_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: FAILED The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, pth):
    '''
    test perform eda function
    '''
    test_df = cls.import_data(pth)
    # call the function under test
    perform_eda(test_df)

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
        logging.error(
            "Testing test_eda: FAILED Some files were not found or empty %s",
            list(
                filter(
                    lambda x: not os.path.isfile(x),
                    file_names)))
        raise err


def test_encoder_helper(encoder_helper, test_df, category_lst, y_field):
    '''
    test encoder helper
    '''
    encoded_df = encoder_helper(test_df, category_lst, y_field)
    try:
        for cat in category_lst:
            assert encoded_df.groupby(cat).mean(
            )[cat + '_' + y_field].equals(test_df.groupby(cat).mean()[y_field])
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: FAILED")
        raise err


def test_perform_feature_engineering(
        perform_feature_engineering, test_df, keep_cols, y_field):
    '''
    test perform_feature_engineering
    '''
    try:
        assert test_df.shape[0] > 0
        assert test_df.shape[1] > 0
        n_rows = test_df.shape[0]
        x_train, x_test, y_train, y_test, x_data = perform_feature_engineering(
            test_df, keep_cols, y_field)
    except AssertionError as err:
        logging.error(
            "Testing import_data: FAILED The file doesn't appear to have rows and columns")
        raise err

    try:
        assert x_train.shape == (int(n_rows - (n_rows * .3)), len(keep_cols))
        assert x_test.shape == (int(n_rows * .3), len(keep_cols))
        assert y_train.shape == (int(n_rows - (n_rows * .3)), )
        assert y_test.shape == (int(n_rows * .3), )
        assert x_data.shape == (n_rows, len(keep_cols))
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: FAILED")
        raise err


def test_train_models(train_models, pth, keep_cols, params, y_field):
    '''
    test train_models
    '''
    test_df = cls.import_data(pth)

    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    encoded_df = cls.encoder_helper(test_df, category_list, y_field)

    x_train, x_test, y_train, y_test, _ = cls.perform_feature_engineering(
        encoded_df, keep_cols, y_field)

    train_models(x_train, x_test, y_train, y_test, params)
    # set expectations for output files
    file_names = [
        './images/lrc_plot.png',
        './images/rfc&lrc.png',
        './images/explainer.png',
        './models/rfc_model.pkl',
        './models/logistic_model.pkl'
    ]
    try:
        # check if expected files exist
        assert all(os.path.isfile(file_name) for file_name in file_names)

        # check if expected files are not empty
        assert all(os.stat(file_name).st_size >
                   0 for file_name in file_names)

        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: FAILED Some files were not found or empty %s",
            list(
                filter(
                    lambda x: not os.path.isfile(x),
                    file_names)))
        raise err


if __name__ == "__main__":
    # test all functions and report final result
    result = []

# columns used for X
    KEEP_COLS = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

# train models
    PARAMS = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    try:
        test_import(cls.import_data, "data/bank_data.csv")
        result.append(True)
    except Exception as e:
        logging.error(e)
        result.append(False)

    try:
        test_eda(cls.perform_eda, "data/bank_data.csv")
        result.append(True)
    except Exception as e:
        logging.error(e)
        result.append(False)

    try:
        TEST_DF = pd.DataFrame(
            {'Gender': ['F', 'M', 'M', 'F'], 'Attrition_Flag': [0, 1, 0, 0]})
        test_encoder_helper(cls.encoder_helper, TEST_DF, ['Gender'], 'Churn')
        result.append(True)
    except Exception as e:
        logging.error(e)
        result.append(False)

    try:
        TEST_DF = pd.util.testing.makeDataFrame()
        test_perform_feature_engineering(
            cls.perform_feature_engineering, TEST_DF, ['A', 'C'], 'D')
        result.append(True)
    except Exception as e:
        logging.error(e)
        result.append(False)

    try:
        test_train_models(
            cls.train_models,
            "data/bank_data.csv",
            KEEP_COLS,
            PARAMS,
            'Churn')
        result.append(True)
    except Exception as e:
        logging.error(e)
        result.append(False)

    passed_cases = len(list(filter(lambda x: x, result)))
    failed_cases = len(list(filter(lambda x: not x, result)))
    TOTAL_CASES = len(result)

    if all(result):
        # log success as final result
        logging.info("Final Test Result : Success %s/%s",
                     passed_cases, TOTAL_CASES
                     )
    else:
        # log failure as final result
        logging.error("Final Test Result : Failed %s/%s",
                      failed_cases, TOTAL_CASES
                      )
