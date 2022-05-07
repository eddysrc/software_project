"""
This program implements the K-Means++ algorithm.
"""

import pandas as pd
import numpy as np
import sys


# Assert given arguments.
def assert_arguments():
    assert len(sys.argv) == 6 or len(sys.argv) == 5, "Invalid Input!"
    arguments = sys.argv
    k = arguments[1]
    try:
        int(k)
    except Exception:
        raise Exception("Invalid Input!")

    if len(sys.argv) == 6:
        max_iter = arguments[2]
        epsilon = arguments[3]
        try:
            int(max_iter)
            float(epsilon)
        except Exception:
            raise Exception("Invalid Input!")
    else:
        epsilon = arguments[2]
        try:
            float(epsilon)
        except Exception:
            raise Exception("Invalid Input!")

    input_file1 = arguments[-2].split('.')
    input_file2 = arguments[-1].split('.')

    if (input_file1[-1].lower() != "txt" and input_file1[-1].lower() != "csv") or (input_file2[-1].lower() != "txt" and input_file2[-1].lower() != "csv"):
        raise Exception("Invalid Input!")


def calc_min_distance(current_vector, centroid_list):
    min_distance = float("inf")

    for centroid in centroid_list:
        sub_vector = np.subtract(current_vector[1:], centroid[1:])
        square_vector = np.square(sub_vector)
        distance = sum(square_vector)

        if distance < min_distance:
            min_distance = distance

    return min_distance


def kmeans_pp_algo(df_matrix, k):
    i = 1
    centroid_list = list()
    number_of_rows = df_matrix.shape[0]

    # Initiate first centroid
    id_array = df_matrix[:, 0]
    random_index = np.random.choice(id_array)
    centroid_i = df_matrix[int(random_index)]
    centroid_list.append(centroid_i)

    while i != k:
        min_distance_list = list()
        probability_list = list()

        for j in range(number_of_rows):
            current_vector = df_matrix[j]
            min_distance = calc_min_distance(current_vector, centroid_list)
            min_distance_list.append(min_distance)

        distance_list_sum = sum(min_distance_list)
        for j in range(number_of_rows):
            current_min_distance = min_distance_list[j]
            current_prob = current_min_distance/distance_list_sum
            probability_list.append(current_prob)

        i += 1
        random_weighted_index = np.random.choice(range(number_of_rows), p=probability_list)
        centroid_list.append(df_matrix[random_weighted_index])

    return centroid_list


def main():
    assert_arguments()

    if len(sys.argv) == 6:
        program_name, k, max_iter, epsilon, input1_filename, input2_filename = sys.argv
        max_iter = int(max_iter)
    else:
        program_name, k, epsilon, input1_filename, input2_filename = sys.argv
        max_iter = 300

    k = int(k)
    epsilon = float(epsilon)

    # Read file 1
    df_file1 = pd.read_csv(input1_filename, header=None)
    column_number1 = df_file1.shape[1]
    file1_headers = ["ID"] + [str(i) for i in range(column_number1-1)]
    df_file1.columns = file1_headers

    # Read file 2
    df_file2 = pd.read_csv(input2_filename, header=None)
    column_number2 = df_file2.shape[1]
    file2_headers = ["ID"] + [str(i + column_number1 - 1) for i in range(column_number2-1)]
    df_file2.columns = file2_headers

    # Build merged and sorted data frame and numpy matrix
    df_combined = pd.merge(df_file1, df_file2, on="ID")
    df_combined = df_combined.sort_values("ID")
    df_matrix = df_combined.to_numpy()

    np.random.seed(0)

    final_centroids = kmeans_pp_algo(df_matrix, k)

    for centroid in final_centroids:
        print(centroid[0])


if __name__ == "__main__":
    main()
