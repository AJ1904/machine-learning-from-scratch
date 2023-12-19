# machine-learning
This repository contains projects related to machine learning csce 633.

## Linear Regression
The objective is to predict combat points using specific attributes associated with each Pokemon, including stamina, attack value, defense value, capture rate, flee rate, and spawn chance.

### Dataset
The dataset contains:
- Rows representing individual Pokemon samples.
- Columns with Pokemon names, attributes (columns 2-7), and the combat point outcome (column 8).

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
In this section, the aim is to perform data preprocessing on the Hitters dataset and then apply classification algorithms, linear regression, and logistic regression as a machine learning pipeline.

### Data Preprocessing
1. **Download and Read Data**: Utilized pandas to read the CSV file and loaded the dataset.
2. **Data Overview**: Used the `head()` function to print the initial data and provided a short description.
3. **Data Dimensions**: Obtained the shape of the data (number of rows and columns).
4. **Missing Values**: Checked for missing values using `isnull()` and computed the total number of missing values with `.sum()`.
5. **Handling Missing Values**: Dropped rows with missing data using `dropna()` function.
6. **Feature and Label Extraction**: Extracted features and the label ('NewLeague').
7. **One-Hot Encoding for Categorical Features**: Separated numerical and non-numerical columns, applied one-hot encoding to categorical features, and concatenated the transformed data.
8. **Transformation of Output**: Converted the output into numerical format using `.replace()` function.

### Models for Hitters
1. **Prediction**: Split the data into 80% training and 20% testing sets. Trained linear regression and logistic regression models.
2. **Feature Coefficients**: Extracted coefficients for each feature in both models and compared their differences or similarities.
3. **ROC Curve**: Plotted the ROC curve for both models and calculated the area under the curve (AUC) measurements.
4. **Optimal Decision Threshold**: Determined the optimal decision threshold to maximize the F1 score and explained the method used for calculation.
5. **Five-fold Cross-validation**: Applied stratified, five-fold cross-validation to repeat the prediction process and observed any changes in features across folds.
6. **AUROC Metrics**: Provided mean and 95% confidence intervals for AUROCs for each model.
7. **F1 Score Metrics**: Provided mean and 95% confidence intervals for the F1 score for each model.

### Mathematical Derivation
Explained the mathematical derivation used for implementing and training linear and logistic regression models.

## References

- [Hitters Dataset Source](https://github.com/jcrouser/islr-python/blob/master/data/Hitters.csv)


## Regularization

In this section, the focus is on utilizing Ridge and Lasso regression techniques for regularization on the Hitters dataset.

### Data Preparation
1. **Read Hitters Dataset**: Loaded the Hitters dataset into a pandas dataframe.
2. **Data Preprocessing**: Performed preprocessing steps similar to Problem B(i) of Homework 1.

### Data Splitting
3. **Data Splitting**: Split the data into train, validation, and test sets. Created X train, X val, X test, y train, y val, and y test sets.

### Ridge Model Training
4. **Train Ridge Model**: Implemented a function `train_ridge` to train the Ridge model for multiple iterations and alpha values. Utilized the validation set for hyperparameter tuning to find the optimal alpha value that provides the best model performance.

### Lasso Model Training
5. **Train Lasso Model**: Created a function `train_lasso` to train the Lasso model for multiple iterations and alpha values. Utilized the validation set for hyperparameter tuning to find the optimal alpha value that provides the best model performance.

### Coefficients Comparison
6. **Ridge Coefficients**: Implemented a function `ridge_coefficients` to obtain the trained Ridge model and its coefficients.
7. **Lasso Coefficients**: Implemented a function `lasso_coefficients` to obtain the trained Lasso model and its coefficients. Compared coefficients between the Lasso and Ridge models.

### Model Evaluation
8. **ROC Curve - Ridge Model**: Implemented a function `ridge_area_under_curve` to calculate area under the curve (AUC) measurements for the Ridge Model. Plotted the ROC curve with appropriate labels, legend, and title.
9. **ROC Curve - Lasso Model**: Implemented a function `lasso_area_under_curve` to calculate area under the curve (AUC) measurements for the Lasso Model. Plotted the ROC curve with appropriate labels, legend, and title.


## References

- [Hitters Dataset Source](https://github.com/jcrouser/islr-python/blob/master/data/Hitters.csv)


## Decision Trees

In this section, the goal is to design and code regression and classification trees using a recursive approach, demonstrating understanding of tree-based algorithms and recursion.

### Maximum-Depth Regression Tree
1. **Implementation of Regression Tree**: Developed a script to grow a maximum-depth regression tree from scratch.
   - Utilized recursion and tree-based algorithms to construct the tree.
   - Generated a tree with the specified maximum depth.

### Two-Class Classification Tree
2. **Implementation of Classification Tree**: Designed a script to grow a two-class classification tree from scratch.
   - Employed similar recursive approaches as in the regression tree to build the classification tree.
   - Constructed the tree for two-class classification.

## Gradient Boosting
In this section, the aim is to utilize XGBoost, an ensemble decision-tree-based machine learning algorithm, with L2 regularization on the Hitters dataset.

### XGBoost Model Training with L2 Regularization
1. **Training XGBoost Model**: Implemented a function `train_XGB` to use XGBoost with L2 regularization for the Hitters dataset.
   - Tuned hyperparameters, including alpha values, using repeated training for optimal performance.
   - Computed AUC values with the validation set to identify the best alpha value.

### Model Training and Testing
2. **Training and Testing the Best Model**: Utilized the best parameters found in step C-i to train and test the XGBoost model on the dataset.

### Model Evaluation - ROC Curve
3. **Plotting ROC Curve**: Implemented a function to plot the ROC curve for the XGBoost model.
   - Calculated area under the curve (AUC) measurements.
   - Included axes labels, legend, and title in the plot for clarity.

### Results Comparison
4. **Comparing XGBoost with Ridge and Lasso Models**: Compared the performance of the XGBoost model with the Ridge and Lasso models from Part A.
   - Reported findings based on the comparison.

## Feed Forward Neural Networks and Convolutional Neural Networks
The task involves processing images from the CIFAR-10 dataset, containing 32x32 color images categorized into 10 classes. The goal is to perform image classification using FNNs and CNNs, evaluating their performance based on different hyperparameters and techniques.


### Image Visualization
1. **Visualization**: Randomly selected and visualized 5-6 images from the dataset.

### Data Exploration
2. **Data Exploration**: Counted the number of samples per class in the training data.


### FNN Image Classification
3. **FNN Classification**: Utilized FNNs for image classification.
   - Experimented with different FNN hyperparameters on a validation set.
   - Monitored loss, reported classification accuracy, training time, and number of learned parameters for each FNN setup.

### Validation Set Experimentation
- Experimented with different hyperparameter combinations (e.g., # layers, # nodes per layer, activation function, dropout, weight regularization) and analyzed their impact.

### Testing the Best Model
- Evaluated the best FNN model on the testing set, reporting classification accuracy and confusion matrix.


### CNN Image Classification
4. **CNN Classification**: Employed CNNs for image classification.
   - Experimented with various CNN hyperparameters on a validation set.
   - Monitored loss, reported classification accuracy, training time, and number of learned parameters for each CNN setup.

### Validation Set Experimentation
- Experimented with different CNN hyperparameter combinations and compared performance metrics with FNNs.

### Testing the Best Model
- Tested the best CNN model on the testing set, comparing its accuracy with the FNN model.


### Bayesian Optimization
5. **Bayesian Optimization**: Utilized Bayesian optimization for hyperparameter tuning and reported classification accuracy on the testing set.

### Fine-tuning Pre-trained CNNs
6. **Fine-tuning**: Explored fine-tuning pre-trained CNNs on CIFAR-10 data with different hyperparameters on a validation set. Reported classification accuracy on both validation and testing sets.

## Support Vector Machines
The task involves preprocessing data, initializing SVM models with different kernels, performing feature selection, training SVM models, visualizing decision boundaries, analyzing results, and considering outlier detection using one-class SVM.

### Data Preprocessing
1. **Data Preprocessing**: Created a binary label based on the "Chance of Admit" column.
   - Converted values larger than the mean to 1 and 0 otherwise.

### Model Initialization
2. **Model Initialization**: Initialized 4 different SVM models with the following kernels:
   - SVC with linear kernel
   - LinearSVC (linear kernel)
   - SVC with RBF kernel
   - SVC with polynomial (degree 3) kernel

### Feature Selection and Model Training
3. **Feature Selection and Model Training**: Trained each SVM model with various feature combinations to predict admission.
   - Explored combinations such as CGPA and SOP, CGPA and GRE Score, SOP with LOR, and LOR with GRE Score.

### Result Visualization
4. **Result Visualization**: Visualized the decision boundary for each model and input combination.

### Result Analysis
5. **Result Analysis**: Analyzed the figures generated to identify the best feature + kernel combinations based on visual inspection.

### Result Postprocessing
6. **Result Postprocessing**: Discussed the possibility of outliers in the data and proposed using a one-class SVM for outlier detection.


## Unsupervised Machine Learning
The goal is to conduct customer segmentation using K-Means clustering and Gaussian Mixture Models (GMMs) on credit card holder data. The data includes various customer behavior measures over a 6-month period for approximately 9000 active credit card holders.

### Histograms
1. **Histograms**: Plotted histograms for variables 2-18 in the data, providing intuitive insights into the distribution of each variable.

### Pearson's Correlation
2. **Pearson's Correlation**: Computed Pearson’s correlation between variables 2-18, visualized the correlation matrix using a heatmap, and discussed potential associations between variables.


### K-Means Clustering
3. **K-Means Clustering**: Utilized K-Means clustering algorithm on variables 2-16.
   - Experimented with different numbers of clusters (K) and identified the optimal K using the elbow method.
   - Reported user assignments to each cluster, centroids, scatter, and discussed findings in relation to users’ percent of full payment and tenure of credit card service.

### K-Means with Feature Selection
4. **K-Means Clustering (Feature Selection)**: Repeated K-Means clustering with a different combination of features informed by previous findings.
   - Discussed the impact of the selected features on clustering results.


### Gaussian Mixture Models
5. **Gaussian Mixture Models**: Employed GMMs to cluster participants based on selected features from previous findings.
   - Reported mean vector, covariance matrix for each Gaussian, and discussed findings.
   - Computed log-likelihood of each sample belonging to the GMM and visualized the histogram of resulting log-likelihood values.

