# Feature Engineering: NBA Player Longevity Prediction

## Project Overview
In this project, I engineered features from an NBA player performance dataset to prepare it for machine learning. The goal of the overarching model is to predict whether an NBA rookie will play in the league for 5 years or more (`target_5yrs`). 

By cleaning the data, removing noise, handling multicollinearity, and engineering new composite metrics, this project transforms raw sports statistics into a robust, model-ready dataset.

## Dataset Source
The dataset used is `nba-players.csv`, which contains various performance metrics for NBA rookies (e.g., points, assists, rebounds, minutes played, and shooting percentages).

## Feature Engineering Workflow
1. **Target Isolation:** Separated the `target_5yrs` variable as the dependent variable for predictive modeling.
2. **Noise Reduction:** Dropped non-predictive string columns (like player names) to prevent data leakage and model bias.
3. **Missing Value Handling:** Identified missing values (commonly in the 3-point percentage column due to 0 attempts) and imputed them appropriately with `0.0` to maintain dataset integrity.
4. **Correlation Analysis:** Analyzed multicollinearity using a Seaborn heatmap. Dropped highly redundant features (like `fgm` and `fga`) in favor of keeping the underlying percentage and the overarching `pts` metric.
5. **Feature Creation:** Engineered a new composite metric called `Efficiency_Per_Min`, which combines Points, Rebounds, and Assists divided by Minutes Played to capture a player's impact independent of their court time.

## Environment Setup
To run this notebook, ensure you have Python installed with the following libraries:
`pip install pandas numpy matplotlib seaborn`
