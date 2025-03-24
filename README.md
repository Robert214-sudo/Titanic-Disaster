# Titanic-Disaster : Machine Learning
This project is a predictive modeling exercise based on the classic Titanic: Machine Learning from Disaster dataset provided by Kaggle. The goal is to build a model that predicts whether a passenger survived the Titanic shipwreck based on features such as age, sex, class, and fare.
## Basic Information
* Person or organization developing model: Robert Bhero, robert.bhero@gwu.edu 
* Model date: February 2025
* Model version: 1.0
* License: MIT
* Model implementation:
## Intended Use
* Primary intended uses: This model is an educational example for random forest regression model for predicting survivors of titanic based on different income, sex etc. It is intended to demonstrate the application of random forest model in data analysis and predictive modeling
* Primary intended users: Students and those that want to learn about modeling
* Out of scope use cases: Any use beyond an educational example is out of scope
## Training Data
* Data dictionary


| Column Name      | Description                                                                     |
|------------------|---------------------------------------------------------------------------------|
| `PassengerId`    | Unique identifier for each passenger                                            |
| `Survived`       | Survival status (Target variable): `0 = No`, `1 = Yes`                          |
| `Pclass`         | Ticket class (proxy for socio-economic status): `1 = 1st`, `2 = 2nd`, `3 = 3rd` |
| `Name`           | Full name of the passenger (includes titles like Mr., Mrs., etc.)               |
| `Sex`            | Gender of the passenger: `male`, `female`                                       |
| `Age`            | Age of the passenger in years. Some values are missing                          |
| `SibSp`          | Number of siblings or spouses aboard the Titanic                                |
| `Parch`          | Number of parents or children aboard the Titanic                                |
| `Ticket`         | Ticket number                                                                   |
| `Fare`           | Passenger fare (numeric – could vary based on class, age, etc.)                 |
| `Cabin`          | Cabin number (many values are missing)                                          |
| `Embarked`       | Port of Embarkation: `C = Cherbourg`, `Q = Queenstown`, `S = Southampton`       |

* Source of training data: Kaggle Titanic Machine Learning Disaster robert.bhero@gwu.edu for more informatio
* How training data was divided into training and validation data: 80% training, 20% validation
* * Number of rows in training and validation data:
   * Training rows: 712 rows
   * Validation rows: 219 rows
## Test data
* Source of test data: Kaggle Titanic Machine Learning Disaster, email robert.bhero@gwu.edu
* Number of rows in test data: 418 rows
## Difference in Columns Between Training and Test Data

The training dataset includes the column `Survived`, which is the **target variable** indicating whether a passenger survived (`1`) or not (`0`).
The **test dataset** does **not** include the `Survived` column, as it is intended for generating predictions to be submitted and evaluated.
## Model Details
### Columns Used as Inputs in the Final Model

The following features were selected and used as inputs to train the final machine learning models:

- `Pclass` – Passenger class (1st, 2nd, 3rd)
- `Sex` – Encoded as 0 (male) and 1 (female)
- `Age` – Age of the passenger (with missing values filled using the median)
- `SibSp` – Number of siblings or spouses aboard
- `Parch` – Number of parents or children aboard
- `Fare` – Passenger fare (missing values filled with the median)
- `Embarked` – Port of embarkation, encoded as: `S = 0`, `C = 1`, `Q = 2`

* These features were selected based on their relevance and contribution to model performance. Categorical features were label-encoded, and numerical features were standardized using `StandardScaler` for consistent scaling across models.
### Model Overview

**Column used as target in the final model:** `Survived`
**Model Type:** Random Forest Classifier
**Software Used:** Python with scikit-learn
**scikit-learn Version:** 1.3.0 
**Hyperparameters:**
  - `n_estimators`: 100
  - `max_depth`: 5
  - `random_state`: 42

### Quantitative Analysis

The model was assessed using the following evaluation metrics:

- **Validation Accuracy Score** – Measured on a hold-out validation set (20% of the training data)
- **Cross-Validation Accuracy** – 5-fold stratified cross-validation to evaluate generalizability
- **Confusion Matrix** – Visualized true vs. predicted values
- **Classification Report** – Precision, recall, F1-score, and support for each class
- **Feature Importance** – Visualized to interpret the impact of each input variable

* These metrics helped assess the model’s predictive performance and ensure that it generalizes well to unseen data.

##  Ethical Considerations in Using Predictive Models for Survival Prediction

The use of predictive models in life or death scenarios—such as survival prediction in disasters raises important ethical considerations. Although the Titanic dataset is a historical case used for educational purposes, it mimics real world decision making where predictive algorithms may be used in fields like healthcare, disaster response, or criminal justice. This section highlights potential limitations, risks and biases associated with survival prediction models, especially when applied beyond academic use.

### Potential Negative Impacts of the Model

Predictive models rely heavily on historical data. If the training data reflects systemic biases or incomplete information, the model may reinforce existing inequalities. For instance, in the Titanic dataset:

- **Gender** and **class (`Pclass`)** are strong predictors of survival. A model may learn to associate higher survival chances with **wealth** or **being female**, reflecting historical survival priorities not necessarily ethical decision-making in modern contexts.
- Using such a model uncritically could normalize these biases and potentially **devalue individuals** from lower social classes or minority groups in serious applications (e.g., disaster triage, emergency planning).

Moreover, if such a model were applied in real world scenarios, it could misinform decision-making processes if the data used is **too narrow, outdated, or demographically skewed**.

*As Noble (2018) explains in* Algorithms of Oppression, *machine learning models trained on biased datasets can unintentionally reinforce structural inequalities.*
## Limitations of the Project

## Only includes a limited set of features
* Doesn't include deep feature engineering (e.g., title extraction, family groups)
* Trained on historical data (biased and limited)
* Performance may vary with different preprocessing strategies

### 📚 What I Learned

- How to clean and preprocess real world data
- How to build classification models using Random Forest and XGBoost
- How to apply cross-validation and interpret performance metrics
- How to assess feature importance and visualize model behavior
- How to think critically about fairness, bias, and ethical implications in AI

###  Future Improvements

- Tune hyperparameters using GridSearchCV or RandomizedSearchCV
- Use feature engineering (e.g., extract titles from names, create family size feature)
- Add fairness metrics to measure bias (e.g., demographic parity, equal opportunity)
- Compare more models (SVM, KNN, Neural Nets)
- Build a simple Streamlit or Flask app for user interaction

### Final Notes

This project provided a comprehensive learning experience in predictive modeling, data ethics, and machine learning workflows. It emphasizes the importance of not just building accurate models but ensuring they are fair, interpretable, and responsibly applied.












