# Customer-Segmentation-using-Kmeans-Clustering-
Unsupervised Machine Learning Project

What is clustering?
- Clustering is a fundamental technique in data analysis and machine learning that involves grouping similar data points based on their characteristics.

What is Kmeans Clustering?
- K-means clustering is a popular unsupervised machine learning algorithm used for partitioning a dataset into K distinct, non-overlapping subsets or clusters.

Key Concepts:
1. Cluster: A group of similar data points.
2. Centroid: The center of a cluster.
3. Distance Metric: A measure of how similar or different data points are. In K-means, Euclidean distance is commonly used.

Steps in Kmean Clustering:
1. Choose the number of clusters (K): This is the number of groups you want to divide your data into. This value is chosen based on prior knowledge or by experimenting.
2. Initialize Centroids: Randomly select K data points from the dataset as the initial centroids. These centroids are the starting points for each cluster.
3. Assign Clusters: Each data point is assigned to the nearest centroid. This step uses the distance metric to determine the nearest centroid.
4. Update Centroids: After all data points are assigned to clusters, recalculate the centroids as the mean of all data points in each cluster.
5. Repeat Steps 3-4: Continue reassigning data points and updating centroids until the centroids no longer change significantly, or a maximum number of iterations is reached.

What is Elbow point method in K-means clustering?
- The Elbow Method is a visual approach used to determine the ideal ‘K’ (number of clusters) in K-means clustering. It operates by calculating the Within-Cluster Sum of Squares (WCSS), which is the total of the squared distances between data points and their cluster center.

Example:
Step 1: Choose K
Suppose we want to cluster a dataset into 2 clusters (K=2).
Step 2: Initialize Centroids
Assume our dataset consists of the following points in a 2D space: (1,1),(2,1),(4,3),(5,4)
We randomly select (1, 1) and (5, 4) as the initial centroids.
<img width="461" alt="image" src="https://github.com/NirajanRijal/Customer-Segmentation-using-Kmeans-Clustering-/assets/160163175/fb0c96a4-34d2-44cc-b847-9c4d13761b37">
For above image, to calculate distance, we use Euclidean distance:
![image](https://github.com/NirajanRijal/Customer-Segmentation-using-Kmeans-Clustering-/assets/160163175/16dba202-99b6-4a99-a0d9-97240b2bc44a)
Step 4: Update Centroids
Calculate the new centroids based on the current cluster assignments.
- Cluster 1: Mean of (1,1) and (2,1) is (1.5, 1)
- Cluster 2: Mean of (4,3) and (5,4) is (4.5, 3.5)
Step 5: Repeat Steps 3-4
Reassign clusters based on new centroids and update centroids again. Repeat until convergence.
