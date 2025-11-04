import random

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    return distance**0.5

def calculate_centroid(points):
    """Calculates the mean of a list of points."""
    if not points:
        return []
    num_dimensions = len(points[0])
    centroid = [0] * num_dimensions
    for point in points:
        for i in range(num_dimensions):
            centroid[i] += point[i]
    return [coord / len(points) for coord in centroid]

def kmeans(data, k, max_iterations=100):
    """Implements K-Means clustering from scratch."""
    # 1. Initialize Centroids
    centroids = random.sample(data, k)

    for _ in range(max_iterations):
        # 2. Assign Data Points to Clusters
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid_index = distances.index(min(distances))
            clusters[closest_centroid_index].append(point)

        # 3. Update Centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:  # Avoid empty clusters
                new_centroids.append(calculate_centroid(cluster))
            else:
                # Handle empty cluster: re-initialize or keep old centroid
                new_centroids.append(random.choice(data)) # Simple re-initialization

        # Check for convergence
        if all(euclidean_distance(old, new) < 1e-6 for old, new in zip(centroids, new_centroids)):
            break
        centroids = new_centroids

    return clusters, centroids

# Example Usage:
data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]
k = 2
clusters, final_centroids = kmeans(data, k)
print("Clusters:", clusters)
print("Final Centroids:", final_centroids)