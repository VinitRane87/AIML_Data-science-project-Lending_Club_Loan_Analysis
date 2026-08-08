# Lending Club Loan Data Analysis

**Deep Learning with Keras and TensorFlow — Course-End Project**

## Problem Statement

For companies like Lending Club, correctly predicting whether a loan will default is critical. This project builds a deep learning model to predict the chance of default for future loans, using historical data from 2007 to 2015. The dataset is imbalanced and includes a range of borrower and loan features that make the problem challenging.

**Domain:** Finance
**Objective:** Create a model that predicts whether or not a loan will default using historical data.

## Data

`lending_club_loan_data.csv` - 9,578 loans with 14 features: credit policy match, loan purpose, interest rate, installment amount, log annual income, debt-to-income ratio, FICO score, credit line history, revolving balance/utilization, recent credit inquiries, delinquencies, public records, and the target `not.fully.paid` (0 = fully paid, 1 = not fully paid / defaulted). About 16% of loans in this dataset were not fully paid.

## Approach

The notebook (`Lending_Club_Loan_Analysis.ipynb`) follows the four required tasks:

1. **Feature Transformation** - the single categorical column (`purpose`) was one-hot encoded into discrete numerical features, since it has no natural order.
2. **Exploratory Data Analysis** - examined the class balance, distributions of key numeric factors (FICO, interest rate, DTI, revolving utilization, etc.), and how these factors relate to default status. Found that lower FICO scores, higher interest rates, and more recent credit inquiries are all associated with a higher chance of default, and that small-business loans and loans that didn't meet LendingClub's own credit-policy criteria default at higher rates.
3. **Additional Feature Engineering** - checked pairwise feature correlations and found one strongly correlated pair (`fico` and `int.rate`, r = -0.72), which makes sense since LendingClub prices interest rate based on credit risk. Dropped the feature with the weaker individual correlation to the target to reduce redundancy before modeling.
4. **Modeling** - built a deep learning model with Keras (TensorFlow backend): a compact neural network (64 -> 32 -> 16 -> 1) with batch normalization, dropout, and class-weighted training to address the class imbalance. Evaluated using AUC, Precision, Recall, ROC and Precision-Recall curves, and a confusion matrix, rather than raw accuracy, since accuracy alone is misleading on an imbalanced target.

## Key Results

- Test AUC: ~0.68
- Recall on the "Not Fully Paid" class: ~66% (catching roughly two-thirds of true defaults)
- Accuracy alone is not a reliable metric here, since ~84% accuracy is achievable by predicting "fully paid" for every loan.

## Files

| File | Description |
|---|---|
| `Lending_Club_Loan_Analysis.ipynb` | Full Jupyter notebook (code + markdown + charts) |
| `Lending_Club_Loan_Analysis.html` | Static HTML export of the notebook |
| `Lending_Club_Loan_Analysis.pdf` | PDF export of the notebook |
| `lending_club_loan_data.csv` | Source dataset |

## Tools

Python, Pandas, NumPy, Scikit-learn (StandardScaler, class_weight), TensorFlow/Keras, Matplotlib, Seaborn, JupyterLab.
