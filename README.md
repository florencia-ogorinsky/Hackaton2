# Airbnb Price Comparison Tool

## Project Overview

In this project, we aimed to develop a price comparison tool for Airbnb listings, addressing the challenges of price variability and information overload for both users and hosts. Trying to find the best features that would help us train our model and predict the prices, we faced some issues, like the correlation between them were not so strong. So, we decided to split our data into subgroups by location. Airbnb prices are heavily influenced by location. Grouping neighborhoods allows for analysis of price differences between distinct areas. This approach significantly improved our prediction accuracy according to the location in the US.

## Problem Statement

Airbnb pricing is highly variable and depends on numerous factors, making it difficult for users to determine fair prices and for hosts to set competitive rates. This project addresses the need for a tool that simplifies price comparisons and provides insights into price determinants.

## Methodology

1.  **Data Acquisition:**
    * Downloaded the Airbnb Open Data dataset from Kaggle using `kagglehub`.
2.  **Data Cleaning and Preprocessing:**
    * Loaded the CSV data using `pandas`.
    * Sampled 5% of the data for efficiency.
    * Inspected the data structure and identified missing values.
    * Removed columns with high missing values ('house_rules', 'license').
    * Removed rows with missing latitude and longitude.
    * Extracted currency and numerical values from 'price' and 'service fee' columns.
    * Handled missing numerical values by imputing with the median.
    * Handled missing categorical values by imputing with the mode.
    * Converted 'last review' to datetime and handled missing dates.
    * Removed duplicate rows.
    * Converted boolean 'instant_bookable' to integers.
    * Combined 'price' and 'service fee' into 'total_price'.
    * Transformed categorical variables ('host_identity_verified', 'neighbourhood group', 'cancellation_policy', 'room type') using one-hot encoding.
3.  **Feature Engineering:**
    * Recognized the strong influence of location on price.
    * Grouped neighborhoods to improve prediction accuracy.
4.  **Model Development:**
    * (Add information about the model you used, training, evaluation, etc.)
    * Developed a predictive model to estimate Airbnb prices based on location and other relevant features.
    * Trained and evaluated the model's performance on the preprocessed dataset.
5.  **Subgroup Analysis:**
    * Divided the dataset into subgroups based on geographical location.
    * Trained and evaluated models for each subgroup to enhance accuracy.

## Key Findings

* Location is a critical determinant of Airbnb prices.
* Grouping neighborhoods significantly improves price prediction accuracy.
* (Add other key findings related to price determinants, e.g., impact of reviews, room type, etc.)

## Technologies Used

* **Python:** Programming language used for data analysis and model development.
* **Pandas:** Data manipulation and analysis.
* **NumPy:** Numerical computations.
* **Scikit-learn:** Machine learning library.
* **Kagglehub:** Dataset download from Kaggle.

## How to Run the Code

1.  Clone the repository to your local machine:
    ```bash
    git clone [repository URL]
    ```
2.  Install the required libraries:
    ```bash
    pip install pandas scikit-learn kagglehub numpy datetime
    ```
3.  Run the main Python script:
    ```bash
    python [script name].py
    ```
    (Provide specific instructions on how to run your code, including any data dependencies)


## Contributors

* Margo Tiamanova
* Florencia Ogorinsky
