# diabetes-prediction-model
Diabetes Prediction with Machine Learning Models

Overview

This project, implemented in the RF.ipynb Jupyter Notebook, focuses on predicting diabetes using a dataset (diabetes_pred.csv) containing 100,000 records with features like age, gender, BMI, and medical history. The notebook performs exploratory data analysis (EDA) and builds machine learning models, including Support Vector Machine (SVM), Random Forest, Decision Tree, and a voting classifier ensemble to predict whether a patient has diabetes.

What's Cool About This Project





Ensemble Learning: The use of a voting classifier that combines predictions from Random Forest, SVM, and Decision Tree models is a standout feature. By leveraging the strengths of multiple algorithms, the ensemble achieves an impressive accuracy of 97.07%, outperforming the standalone SVM (92%).



Dimensionality Reduction with PCA: The SVM model uses Principal Component Analysis (PCA) to reduce the feature space to 2 dimensions, making it computationally efficient and potentially visualizable, which is great for understanding high-dimensional data.



Comprehensive EDA: The notebook includes thorough data exploration with histograms, summary statistics, and checks for missing values, providing a solid foundation for model building.



Real-World Application: Predicting diabetes is a practical use case with significant healthcare implications, showcasing how machine learning can assist in medical diagnostics.



Imbalanced Dataset Handling: The project tackles an imbalanced dataset (91,500 non-diabetic vs. 8,500 diabetic cases), highlighting real-world challenges in classification tasks.

What I Learned





Data Preprocessing: I learned the importance of standardizing features using StandardScaler to ensure consistent scale across variables, which is critical for models like SVM.



Dimensionality Reduction: Using PCA to reduce features to 2D for the SVM model taught me how to handle high-dimensional data efficiently while preserving most of the variance.



Model Evaluation: I gained experience with evaluation metrics like accuracy, precision, recall, F1-score, and confusion matrices, especially in the context of imbalanced datasets where recall for the minority class (diabetes) is critical.



Ensemble Techniques: Implementing a voting classifier showed me how combining multiple models can improve performance by leveraging their complementary strengths.



Class Imbalance Awareness: The lower recall (70%) for the diabetic class highlighted the challenges of imbalanced datasets, prompting me to consider techniques like SMOTE or class weighting for future improvements.



Visualization and EDA: Creating histograms and analyzing data distributions helped me understand the dataset's characteristics and identify potential issues like class imbalance.

Dataset





File: diabetes_pred.csv



Size: 100,000 records, 9 columns



Features: gender (object), age (float64), hypertension (int64), heart_disease (int64), smoking_history (object), bmi (float64), HbA1c_level (float64), blood_glucose_level (int64)



Target: diabetes (int64, binary: 0 = no diabetes, 1 = diabetes)



Key Insight: The dataset is imbalanced, with 91,500 non-diabetic and 8,500 diabetic cases.

Project Structure





Data Loading and EDA:





Loads the dataset using pandas.



Checks for missing values (none found).



Generates summary statistics and histograms for numerical features.



Analyzes the target variable distribution.



SVM Model:





Splits data into 70% training and 30% testing sets.



Standardizes features and applies PCA (2 components).



Trains a linear SVM and achieves 92% accuracy.



Ensemble Voting Classifier:





Trains Random Forest, SVM (with PCA), and Decision Tree models.



Combines predictions using majority voting.



Achieves 97.07% accuracy, with a detailed classification report and confusion matrix.

Requirements





Python 3.x



Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, scipy

How to Run





Ensure the required libraries are installed:

pip install pandas numpy seaborn matplotlib scikit-learn scipy



Place diabetes_pred.csv in the same directory as RF.ipynb.



Open RF.ipynb in Jupyter Notebook or JupyterLab.



Run the cells sequentially to perform EDA and train the models.

Results





SVM Accuracy: 92%



Voting Classifier Accuracy: 97.07%



Classification Report:





Precision (Class 0): 0.97, (Class 1): 0.94



Recall (Class 0): 1.00, (Class 1): 0.70



F1-score (Class 0): 0.98, (Class 1): 0.80



Confusion Matrix:





True Negatives: 27,336



False Positives: 114



False Negatives: 764



True Positives: 1,786

Future Improvements





Handle Class Imbalance: Apply techniques like SMOTE, oversampling, or class weighting to improve recall for the diabetic class.



Hyperparameter Tuning: Use GridSearchCV to optimize parameters for Random Forest, SVM, and Decision Tree models.



Feature Engineering: Encode categorical variables (gender, smoking_history) using one-hot encoding or other methods to include them in the models.



Visualization: Plot PCA-transformed data or decision boundaries to visualize model behavior.



Cross-Validation: Implement k-fold cross-validation to ensure robust performance metrics.
