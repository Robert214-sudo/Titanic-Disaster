# Titanic-Disaster
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

* Source of training data: Kaggle Titanic Machine Learning Disaster robert.bhero@gwu.edu for more information
* How training data was divided into training and validation data: 80% training, 20% validation
* * Number of rows in training and validation data:
   * Training rows: 712 rows
   * Validation rows: 219 rows


