# Airbnb Price Comparison Tool

## 📌 Overview
**Airbnb Price Comparison Tool** is a data analysis project focused on developing a tool to estimate and compare Airbnb listing prices. The goal is to analyze various factors influencing Airbnb pricing and provide insights to both users and hosts for better decision-making.

The dataset includes attributes such as location, room type, amenities, reviews, and other features to predict and analyze Airbnb listing prices.

## ❓ Problem It Solves
Airbnb pricing is highly variable and depends on numerous factors, making it challenging for users to determine fair prices and for hosts to set competitive rates. This project enables:
- Accurate price estimation based on listing attributes.
- Price comparison across different locations and listing types.
- Identification of key factors influencing Airbnb prices.
- Data-driven insights to optimize pricing strategies for hosts.

## 🚀 Features
✅ **Data Preprocessing**: Handles missing data, removes duplicates, and transforms categorical variables to ensure clean and usable data.

✅ **Feature Engineering**: Creates new features and groups neighborhoods to improve prediction accuracy

✅ **Correlation Analysis**: Identifies relationships between listing attributes and price

✅ **Subgroup Analysis**: Analyzes subgroups based on location to improve model accuracy

✅ **Predictive Modeling**: Develops a model to estimate Airbnb prices based on relevant features

✅ **Visualization**: Provides visualizations to understand price distributions and feature impacts

## 🛠 How to Run
### Prerequisites
Ensure you have:
- **Python 3.x** installed
- **Jupyter Notebook** or a Python interpreter (e.g., VS Code)
- Required libraries: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `kagglehub`, `datetime`

### Installation & Execution
1️⃣ **Clone the Repository**:

git clone [https://github.com/florencia-ogorinsky/Hackaton2.git](https://github.com/florencia-ogorinsky/Hackaton2.git)


2️⃣ **Install Dependencies**:
pip install pandas numpy scipy matplotlib seaborn scikit-learn kagglehub datetime


3️⃣ **Run the Analysis**:
-Open Jupyter Notebook
-Execute code blocks sequentially to analyze data and generate insights.


4️⃣ **Dataset**:
The dataset is downloaded from Kaggle using kagglehub.
[🔗 Kaggle]([[https://www.kaggle.com]](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata)


## 📊 **Understanding the Data**
The dataset includes various Airbnb listing attributes, such as:

Price: Listing price
* Neighborhood and neighborhood group
* Private room, entire home/apt, shared room.
* Number of reviews and last review date.
* Available amenities
* Cancellation policy for the listing
* Host verification status
... and many more


## **Conclusion** 📌

Our results show that looking at different neighborhoods separately helps us predict Airbnb prices better. Each area has its own price patterns, so splitting the data makes the predictions more accurate. We also used a special math trick (log transformation) to make prices easier to compare, helping our model work better.

## 🚀 **How This Can Help**:

✅  Hosts can see how much similar homes cost and set fair prices
✅  Guests can find good deals and compare prices more easily
✅  Market Experts can study price trends and see which areas are growing
✅  Real Estate Agents can use the data to find good places for investment

## 🔧 **How to Make It Even Better**:

🔹 More Details: Adding things like seasons, extra features in the house, and reviews could improve predictions.🔹 Smarter Models: Trying out advanced AI models may give even better results.🔹 Time-Based Pricing: Looking at how prices change over time can help make smarter predictions.🔹 More Location Info: Adding data on crime rates, attractions, and local events could improve accuracy.

With these improvements, our tool could become even more useful for Airbnb users, hosts, and business experts.


⭐ **Contributions & Feedback are Welcome**!
📩 Feel free to open issues or pull requests to enhance the project.

📌 Authors: Margo Tiamanova & Florencia Ogorinsky




