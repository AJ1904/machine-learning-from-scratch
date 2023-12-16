# machine-learning
This repository contains projects related to machine learning csce 633.

## Linear Regression
The objective is to predict combat points using specific attributes associated with each Pokemon, including stamina, attack value, defense value, capture rate, flee rate, and spawn chance.

### Dataset
The dataset contains:
- Rows representing individual Pokemon samples.
- Columns with Pokemon names, attributes (columns 2-7), and the combat point outcome (column 8).

### Steps Taken
#### 1. Data Exploration
- Conducted exploratory data analysis.
- Plotted 2-D scatter plots.
- Computed Pearson’s correlation coefficient between features and combat points to identify the most predictive attributes.

#### 2. Feature Correlation
- Explored feature-to-feature relationships.
- Plotted 2-D scatter plots.
- Computed Pearson’s correlation coefficient between features to identify correlated attributes.

#### 3. Linear Regression Implementation
- Implemented linear regression using ordinary least squares (OLS).
- Split the data into 5 folds for cross-validation.
- Computed the square root of the residual sum of squares error (RSS) for each fold and averaged across all folds.

#### 4. Feature Combination Experimentation
- Experimented with different feature combinations based on findings from data exploration and feature correlation.
- Reported results for each combination tested.

#### 5. Mathematical Derivation
- Detailed the mathematical derivation for implementing and training the linear regression model with OLS solution.

### Implementation Details

- Python code was utilized for data analysis and model implementation.
- Libraries like NumPy, pandas, and matplotlib were used for data manipulation, computation, and visualization.
- The linear regression algorithm and 5-fold cross-validation were manually implemented.

### References

- [Link to the Pokemon GO study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5174727/)
- [Wikipedia - Pearson's Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)



## Logistic Regression

## Regularization

## Decision Trees

## Gradient Boosting

## Feed Forward Neural Networks

## Convolutional Neural Networks

## Support Vector Machines

## Unsupervised Machine Learning
