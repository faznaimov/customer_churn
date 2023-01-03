'''
Package that hold functions transfered from jupyter notebook.

Author: Faz Naimov
Date: 1/1/2023
'''

# import libraries
import warnings
import os
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap

warnings.filterwarnings("ignore")
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    '''
    data = pd.read_csv(pth)

    return data


def perform_eda(bank_df):
    '''
    perform eda on df and save figures to images folder
    input:
            bank_df: pandas dataframe

    output:
            None
    '''

    # churn histogram
    plt.figure(figsize=(20, 10))
    bank_df['Churn'] = bank_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.title('Attrited and Existing Customer Distribution')
    plt.xlabel('Churn')
    bank_df['Churn'].hist()
    plt.tight_layout()
    plt.savefig('./images/churn.png')

    # customer age
    plt.figure(figsize=(20, 10))
    bank_df['Customer_Age'].hist()
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.tight_layout()
    plt.savefig('./images/age_dist.png')

    # marital status
    plt.figure(figsize=(20, 10))
    bank_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Marital Status Barchart')
    plt.xlabel('Marital Status')
    plt.tight_layout()
    plt.savefig('./images/marital_status.png')

    # Total_Trans_Ct
    plt.figure(figsize=(20, 10))
    sns_plot = sns.histplot(bank_df['Total_Trans_Ct'], stat='density', kde=True)
    plt.tight_layout()
    sns_plot.figure.savefig('./images/total_trans_ct.png')

    # Corr
    plt.figure(figsize=(20, 12))
    sns_plot = sns.heatmap(
        bank_df.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.tight_layout()
    sns_plot.figure.savefig('./images/corr.png')


def encoder_helper(bank_df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            bank_df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    '''
    for cat in category_lst:
        bank_df[cat + '_Churn'] = bank_df[cat].map(bank_df.groupby(cat).mean()['Churn'])
    return bank_df


def perform_feature_engineering(bank_df, keep_cols):
    '''
    input:
              bank_df: pandas dataframe
              keep_cols: list of columns that model is trained on

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    y_data = bank_df['Churn']
    x_data = pd.DataFrame()
    x_data[keep_cols] = bank_df[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test, x_data


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                model):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions
            y_test_preds: test predictions
            model: model name
    output:
             None
    '''

    plt.figure('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str(model+' Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str(model+' Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'./images/{model.lower().replace(" ", "_")}.png')


def feature_importance_plot(model, x_data):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(18, 7))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig('./images/feature_importance.png')


def train_models(x_train, x_test, y_train, y_test, param_grid):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
              param_grid: grid search parameters
    output:
              None
    '''

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # plots
    plt.figure(figsize=(5, 5))
    lrc_plot = RocCurveDisplay.from_estimator(lrc, x_test, y_test)
    plt.tight_layout()
    lrc_plot.figure_.savefig('./images/lrc_plot.png')

    plt.figure(figsize=(15, 8))
    rfc_disp = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, x_test, y_test, ax=plt.gca(), alpha=0.8)
    lrc_plot.plot(ax=plt.gca(), alpha=0.8)
    plt.tight_layout()
    rfc_disp.figure_.savefig('./images/rfc&lrc.png')

    plt.figure(figsize=(5, 20))
    shap_values = shap.TreeExplainer(cv_rfc.best_estimator_).shap_values(x_test)
    shap.summary_plot(
        shap_values,
        x_test,
        plot_type="bar",
        show=False)
    plt.tight_layout()
    plt.savefig('./images/explainer.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf, cv_rfc
