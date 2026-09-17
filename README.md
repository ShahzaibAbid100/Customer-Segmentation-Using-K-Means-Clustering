# Customer Segmentation Using K-Means Clustering

## Project Overview

This project uses **K-Means Clustering**, an unsupervised machine learning algorithm, to segment customers into different groups based on their annual income and spending score.

Unlike supervised learning, this project does not use a target variable. Instead, K-Means discovers patterns in the customer data and groups similar customers together.

## Objective

The objective of this project is to:

- Understand K-Means Clustering
- Group customers with similar characteristics
- Use the Elbow Method to select the number of clusters
- Visualize customer segments
- Analyze the characteristics of each cluster

## Dataset

The project uses the **Mall Customers Dataset**.

Important columns include:

- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

For the initial clustering model, the main features used are:

- Annual Income
- Spending Score

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Workflow

### 1. Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

### 2. Load the Dataset

```python
df = pd.read_csv("Mall_Customers.csv")

df.head()
```

### 3. Explore the Dataset

```python
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
```

### 4. Select Features

Annual income and spending score are selected for clustering.

```python
X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]
```

K-Means is an unsupervised learning algorithm, so there is no target variable `y`.

### 5. Visualize the Data

```python
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"]
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customers Before Clustering")

plt.show()
```

### 6. Feature Scaling

StandardScaler is used to put the selected features on a standardized scale.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Feature scaling is useful because K-Means uses distances when assigning data points to clusters.

## Elbow Method

The Elbow Method is used to investigate a suitable number of clusters.

```python
inertia = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)
```

The results are visualized:

```python
plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()
```

The elbow point is where the reduction in inertia begins to slow down significantly.

## Train the K-Means Model

After selecting the number of clusters, the final K-Means model is created.

Example:

```python
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)
```

## Cluster Assignment

K-Means assigns each customer to a cluster.

```python
clusters = kmeans.labels_

df["Cluster"] = clusters
```

The updated dataset can be viewed using:

```python
df.head()
```

## Cluster Visualization

The identified customer groups are visualized using a scatter plot.

```python
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=clusters
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")

plt.show()
```

## Cluster Analysis

The average characteristics of each cluster can be examined using:

```python
cluster_summary = df.groupby("Cluster")[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean()

print(cluster_summary)
```

This helps us understand the characteristics of customers belonging to each cluster.

## Key Concepts Learned

Through this project, I practiced:

- Unsupervised Machine Learning
- K-Means Clustering
- Clusters
- Centroids
- Feature Scaling
- StandardScaler
- Inertia / WCSS
- Elbow Method
- Cluster Visualization
- Customer Segmentation
- Cluster Interpretation

## How K-Means Works

K-Means follows an iterative process:

1. Choose the number of clusters `K`.
2. Initialize `K` centroids.
3. Calculate the distance between data points and centroids.
4. Assign each data point to its nearest centroid.
5. Recalculate each centroid using the points assigned to its cluster.
6. Repeat the assignment and update steps until the clusters stabilize.

## Conclusion

This project demonstrates how **K-Means Clustering** can be used to discover customer segments without predefined class labels.

The Elbow Method helps investigate an appropriate number of clusters, while visualization and cluster-level statistics help interpret the resulting customer groups.

Customer segmentation can help businesses better understand different customer behaviors and support more targeted business analysis.

## Author

**Shahzaib Abid**

Machine Learning & AI Learner
