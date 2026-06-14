# Feature Engineering: NBA Player Longevity Prediction

## Project Overview
In this project, I engineered features from an NBA player performance dataset to prepare it for machine learning. The goal of the model is to predict whether an NBA rookie will play in the league for 5 years or more (`target_5yrs`). By cleaning the data, handling multicollinearity, scaling variables, and engineering composite metrics, this project transforms raw sports statistics into a robust, model-ready dataset.

## Dataset Source
The dataset used is `nba-players.csv`, containing performance metrics for NBA rookies (e.g., points, assists, rebounds, minutes played, and shooting percentages).

## Feature Engineering Workflow
1. **Data Exploration & Class Balance:** Loaded the data and verified the distribution of the `target_5yrs` variable using value counts and visualizations to check for severe class imbalances.
2. **Noise Reduction:** Dropped non-predictive string columns (like player names) to prevent data leakage and model bias.
3. **Missing Value Imputation:** Identified missing values in the 3-point percentage column (caused by 0 attempts) and imputed them with `0.0` to maintain dataset integrity.
4. **Multicollinearity & Correlation:** Analyzed feature relationships using a Seaborn heatmap. Dropped highly redundant features (like `fgm`, `fga`, `ftm`, `fta`) to reduce multicollinearity, retaining the overarching percentage and points metrics.
5. **Feature Extraction:** Engineered a composite metric called `Efficiency_Per_Min` ((Points + Rebounds + Assists) / Minutes). This captures a player's raw impact independent of their court time.
6. **Feature Scaling:** Applied `StandardScaler` to all numerical predictor variables to ensure distance-based machine learning models (like SVM or KNN) evaluate the statistics on a uniform scale.

## Insights & Stakeholder Summary
Based on the correlation analysis and feature extraction, the strongest predictors for a player lasting 5+ years in the NBA are total games played, total minutes, and overall points. By stripping away redundant sub-metrics (like field goals attempted) and engineering the `Efficiency_Per_Min` metric, the resulting dataset provides a clear, mathematically sound foundation for deploying predictive longevity models. 

## Environment Setup
`pip install pandas numpy matplotlib seaborn scikit-learn`
