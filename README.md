# Final Project Report: [NBA Stats Predictor]

**Course**: CS383 - Introduction to Machine Learning  
**Semester**: Fall 2024  
**Team Members**: Ramla Mohammed [Ramla12]  
**Instructor**: Adam Poliak 

---

## Table of Contents
1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Problem Statement](#problem-statement)
4. [Related Work](#related-work)
5. [Data Description](#data-description)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Discussion](#discussion)
9. [Conclusion and Future Work](#conclusion-and-future-work)
10. [References](#references)

---

## Abstract
This project aims to predict NBA player performance based on key game statistics. I explored linear regression and random forest models, analyzing their performance on cleaned and preprocessed player data. Random forests achieved the highest accuracy (86%) and ROC-AUC (94%), demonstrating their ability to handle non-linear relationships in the data. These results align with prior research in basketball analytics, underscoring the importance of statistical modeling in sports performance prediction.


---

## Introduction
Basketball analytics is an increasingly vital field, leveraging large datasets to gain insights into player and team performance. This project investigates the use of machine learning to predict player performance, focusing on NBA game statistics. I apply linear regression and random forest models to analyze predictors such as minutes played, assists, and shooting percentages. Our objective is to determine which features are most influential and to compare the performance of these models. This work builds upon prior research, such as Nguyen et al.'s 2022 study on machine learning in sports analytics.


---

## Problem Statement
The primary objective of this project is to predict whether an NBA player will score above or below a predefined point threshold (I set it to 10 points) based on game statistics. The secondary goal is to evaluate the effectiveness of linear regression versus random forest models for this task. Through this research I hope to contribute to the growing field of sports analytics, with potential applications in player evaluation and team strategy.


---

## Related Work
The study "The Application of Machine Learning and Deep Learning in Sport: Predicting NBA Players’ Performance and Popularity" by Nguyen et al. (2022) explored the use of traditional machine learning and deep learning models to predict NBA players' performance. The authors highlighted the effectiveness of random forests for structured, small datasets, achieving high ROC-AUC and recall scores. Inspired by this work, my project also investigates the performance of linear regression and random forest models on a similar dataset.


---

## Data Description
**Source**: NBA player statistics were sourced from Kaggle.  
**Size and Format**: The dataset contains 625 samples and 7 predictive features, including `Age`, `MP`, `FG%`, `3P%`, `FT%`, `TRB`, and `AST`.  
**Preprocessing**: Missing values were removed, and the target variable (`PTS`) was binarized to classify players scoring above or below a threshold. Data was split into 80% training and 20% testing subsets.

---

## Methodology
Outline your approach, including:
1. The algorithms used are Linear Regression and Random Forests.  
2. The models were trained on an 80-20 train-test split
3. No hyper parameter tuning was employed
4. The libraries used are: scikit-learn, pandas, numpy, seaborn, and matplotlib 

---

## Results
Present the results of your experiments, including:
- Key metrics (e.g., accuracy, precision, recall, F1 score, etc.).
- Comparisons between models or baselines.
- Visualizations (e.g., plots, confusion matrices).

**Example:**

| Model          | Accuracy | Precision | Recall | F1 Score | Roc-AUC
|-----------------|----------|-----------|--------|----------|---------
| Logistic Reg.   | 0.8560    | 0.81      | 0.84   | 0.82    |  0.85  |
| Random Forest   | 0.8640    | 0.85      | 0.80   | 0.82     |  0.94   |

Confusion Matrix: Linear Regression
[[65 10]
 [ 8 42]]

   Feature  Coefficient
4     FT%     3.791178
6     AST     1.508155
1      MP     1.419793
5     TRB     0.564511
3     3P%     0.231156
0     Age    -0.690376
2     FG%    -1.703160

Confusion Matrix: Random Forest
[[68  7]
 [10 40]]

---

## Discussion
Interpret your results:

Both models demonstrated strong performance, with Random Forest slightly outperforming Linear Regression in overall metrics. Linear Regression achieved an accuracy of 85.6%, a precision of 0.81, a recall of 0.84, and an F1-score of 0.82. It demonstrated strong recall, identifying most high-performing players, but slightly struggled with false positives, leading to a lower ROC-AUC of 0.85. Random Forest improved on these results, achieving an accuracy of 86.4%, a precision of 0.85, a recall of 0.80, and an F1-score of 0.82, with a significantly higher ROC-AUC of 0.94. This indicates that Random Forest is better at distinguishing between high and low-scoring players, especially in cases with overlapping features or imbalanced classes. 

The confusion matrices highlight the models' differences. Linear Regression correctly identified 42 high-scoring players but made 10 false positive predictions, overestimating performance for low-scoring players. Random Forest reduced false positives to 7 while maintaining 40 true positives, making it more conservative in its predictions. Feature analysis, based on Linear Regression coefficients, revealed that free throw percentage (FT%) and assists (AST) were the most influential predictors of performance, aligning with basketball analytics principles. Additionally, Linear Regression indicated that field goal percentage (FG%) and age negatively correlated with performance.

In summary, Random Forest proved more effective for this task due to its ability to model non-linear interactions and reduce false positives, while Linear Regression offered more interpretable insights into feature importance. Future work could explore hyperparameter tuning for Random Forest and feature engineering to improve performance further.


---

## Conclusion and Future Work

With more time I would work on getting the feature importance of the random forests model to see how they compare to each other, and I woyld try implementing a Neural network to see how it compares. I would also explore team-level features or contextual data like the opposing teams strength to see how that affects the results.

---

## References

Kaggle. "NBA Players Statistics Dataset." Kaggle.com. Accessed on [12/10/2024]. Available at: [(https://www.kaggle.com/datasets/joebeachcapital/nba-player-statistic)].
Pedregosa, Fabian, et al. "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, vol. 12, 2011, pp. 2825–2830.
Hunter, J. D. "Matplotlib: A 2D Graphics Environment." Computing in Science & Engineering, vol. 9, no. 3, 2007, pp. 90–95.
Smith, John, et al. "The Application of Machine Learning and Deep Learning in Sport: Predicting NBA Players’ Performance and Popularity." [Include publication details].

