"""
Python program that implements k-means algorithm.
"""
import math
import sys


# Class Cluster
class Cluster:
    def __init__(self, dimension):
        self.dimension = dimension
        self.centroid = [0.] * dimension
        self.cluster_vectors = list()

    def set_centroid(self, data_points):
        assert len(data_points) == self.dimension
        self.centroid = data_points


# Assert given arguments.
def assert_arguments():
    assert len(sys.argv) == 5 or len(sys.argv) == 4, "Invalid Input!"
    arguments = sys.argv
    k = arguments[1]
    try:
        int(k)
    except Exception:
        raise Exception("Invalid Input!")

    if len(sys.argv) == 5:
        max_iter = arguments[2]
        try:
            int(max_iter)
        except Exception:
            raise Exception("Invalid Input!")


# Open file, in an array initialize Clusters with empty sets and centroid.
def initialize_clusters(cluster_list, input_filename, k):
    try:
        with open(input_filename) as input_file:
            lines = input_file.readlines()

            assert k < len(lines), "Invalid Input!"

            # Iterate over first K lines
            for i in range(k):
                line = lines[i].split(',')
                current_cluster = Cluster(len(line))
                data_points = list()

                for data_point in line:
                    data_points.append(float(data_point))

                current_cluster.set_centroid(data_points)
                cluster_list.append(current_cluster)
    except Exception:
        raise Exception("An Error Has Occurred")


# Open file and assign all data-points to the clusters based on euclidean norm.
def assign_data_points_to_clusters(file_path, cluster_list):
    # Read file
    with open(file_path) as file:
        vectors = file.readlines()

        for vector_repr in vectors:
            # Initialize vectors
            vector = vector_repr.split(",")
            vector = [float(entry) for entry in vector]

            # Find the closest cluster and add vector
            cluster = find_closest_cluster(vector, cluster_list)
            cluster.cluster_vectors.append(vector)


def find_closest_cluster(vector, cluster_list):
    min_distance = float('inf')
    min_cluster = cluster_list[0]

    for cluster in cluster_list:
        distance = calculate_distance_from_vector(vector, cluster.centroid)
        if distance < min_distance:
            min_distance = distance
            min_cluster = cluster

    return min_cluster


def calculate_distance_from_vector(vector, centroid):
    distance = 0

    for (entry1, entry2) in zip(vector, centroid):
        distance += (entry1 - entry2) ** 2

    return distance


def update_centroids(cluster_list):
    for cluster in cluster_list:
        cluster.set_centroid(compute_centroid(cluster))
        cluster.cluster_vectors = list()


def compute_centroid(cluster):
    # Initialize the centroid
    vectors_sum = [0.] * cluster.dimension

    # Sum the vectors in the cluster to the centroid
    for vector in cluster.cluster_vectors:
        for i, entry in enumerate(vector):
            vectors_sum[i] += entry

    # Cluster is empty
    if len(cluster.cluster_vectors) == 0:
        return cluster.centroid

    # Divide the entries by the amount of vectors in the cluster and format
    for i in range(len(vectors_sum)):
        vectors_sum[i] /= len(cluster.cluster_vectors)

    return vectors_sum


def write_to_file(output_file, cluster_list):
    with open(output_file, 'w') as file:
        for cluster in cluster_list:
            cluster_centroid_string = [str("%.4f" % entry) for entry in cluster.centroid]
            input_text = str.join(',', cluster_centroid_string)
            file.write(input_text + "\n")
        file.write("")


def calc_euclidean_norm(vector):
    vector_squared = [x**2 for x in vector]
    vector_sum = sum(vector_squared)
    return math.sqrt(vector_sum)


def check_euclidean_norms(vectors_delta, epsilon):
    for delta in vectors_delta:
        if calc_euclidean_norm(delta) > epsilon:
            return True
    return False


def subtract_vectors(vector1, vector2):
    return [entry1 - entry2 for entry1, entry2 in zip(vector1, vector2)]


# Update centroids.
def main():
    assert_arguments()

    if len(sys.argv) == 5:
        program_name, k, max_iter, input_filename, output_filename = sys.argv
        max_iter = int(max_iter)
    else:
        program_name, k, input_filename, output_filename = sys.argv
        max_iter = 200

    k = int(k)

    cluster_list = list()
    initialize_clusters(cluster_list, input_filename, k)

    epsilon = 0.001
    iterations = 0
    delta_vectors = [cluster.centroid for cluster in cluster_list]

    while check_euclidean_norms(delta_vectors, epsilon) and iterations < max_iter:

        before_vectors = [cluster.centroid for cluster in cluster_list]
        assign_data_points_to_clusters(input_filename, cluster_list)
        update_centroids(cluster_list)
        after_vectors = [cluster.centroid for cluster in cluster_list]

        delta_vectors = [subtract_vectors(before_vector, after_vector) for before_vector, after_vector in zip(before_vectors, after_vectors)]

        iterations += 1

    write_to_file(output_filename, cluster_list)


if __name__ == "__main__":
    main()
