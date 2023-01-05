# Predict Customer Churn

- Project **Predict Customer Churn** in [ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)  program by Udacity.


## Project Description
The project is on credit card customers that are most likely to churn, based on the dataset available in [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers). This project implements two different churn predictors: a Random Forest Classifier and a Logistic Regression method.

The aim of this project is to demonstrate my competencies in writing clean code, to be specific:

- The ability to write modular and efficient code, with proper documentation and style check (using pylint and autopep8)
- Implement best practices such as errors handling, testing and logging

## Files and data description
The directories structure are list as below:
```bash
.
├── data
│   └── bank_data.csv
├── images
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── churn_notebook.ipynb
├── churn_library.py
├── churn_script.py
├── churn_script_logging_and_tests.py
├── README.md
└── requirements.txt
```
The original given file (`churn_notebook.ipynb`) is refactored into `.py` files for modularity, reusability, as well as ease of testing and logging.

More description on some files and folders included in the project:
- `churn_library.py`: contains utility functions that help with data analysis and model training
- `churn_script.py`: contains script that runs the functions from `churn_library.py`
- `churn_script_logging_and_tests.py`: used for testing functions in churn_library module
- `data`: the directory that contains the data file used in this project
- `images`: used for saving images of data analysis and model analysis
- `logs`: contains log generated from running testing script
- `models`: used for storing model that are ready to use in production


## Running

### Dependencies

List of libraries used for this project:

```
autopep8==2.0.1
joblib==1.2.0
matplotlib==3.6.2
numpy==1.23.5
pandas==1.5.2
pylint==2.15.9
scikit-learn==1.2.0
seaborn==0.12.2
shap==0.41.0
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies from the `requirements.txt`

```
pip install -r requirements.txt
```
### Modeling

Run the following command to execute the main script
```
python churn_script.py
``` 
script execution generates
- EDA and model metrics plots are available in directory ```./images/```
- Model pickle files are available in directory ```./models/```
- Log files are available in directory ```./logs/churn_library.log``` 

### Testing and Logging

Run the following command to run the tests script 
```
python churn_script_logging_and_tests.py
```

script execution generates
- Log file ```./logs/churn_library.log```

## License
Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See ```LICENSE``` for more information.
