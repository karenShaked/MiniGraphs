from typing import List
from collections import Counter
import networkx as nx
import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
ITER = 30


def consolidate_histogram_ranges(histogram, range_step=1000):
    new_histogram = {}
    for key, count in histogram.items():
        range_key = f'{(key // range_step) * range_step}-{((key // range_step) + 1) * range_step}'
        if range_key not in new_histogram:
            new_histogram[range_key] = 0
        new_histogram[range_key] += count
    sorted_keys = sorted(new_histogram, key=lambda x: int(x.split('-')[0]))
    sorted_values = [new_histogram[key] for key in sorted_keys]
    plt.figure(figsize=(12, 8))
    plt.bar(sorted_keys, sorted_values, color='skyblue')
    plt.xlabel('Range of Values')
    plt.ylabel('Count')
    plt.title('Histogram Of Biggest Components Sizes for 30 Random Graphs Of Size 1000000')
    plt.xticks(rotation=45)
    plt.show()


def k_max_connencted_components_sizes(p: float, n: int, k: int):
    k_lst = [[0] for i in range(k)]
    first_histo = Counter()
    for i in range(ITER):
        histo = connected_components_sizes(p, n)
        first_max_key = max(histo.keys())
        first_histo[first_max_key] = first_histo.get(first_max_key, 0) + 1
        for j in range(k):
            # Check if the histogram is empty to avoid errors
            if not histo:
                break

            # Find the max i component size
            max_key = max(histo.keys())

            # Reduce the value by 1
            histo[max_key] -= 1

            # If the value is 0, delete the key
            if histo[max_key] == 0:
                del histo[max_key]
            k_lst[j][0] += max_key
    k_lst_avg = [[element / ITER for element in sublist] for sublist in k_lst]
    return k_lst_avg, first_histo


def calculate_k_connected_components_for_all_n(n_lst: List[int], p_lst, theta_l, k=30):
    connected_comp_sizes = [[] for _ in range(k)]
    for n, p in zip(n_lst, p_lst):
        k_lst_avg, histo_first = k_max_connencted_components_sizes(p, n, k)
        connected_comp_sizes = [previous + new for previous, new in zip(connected_comp_sizes, k_lst_avg)]
        # if n == n_lst[0]:
            # plot_histo(histo_first, n)
        # elif n == n_lst[-1]:
            # consolidate_histogram_ranges(histo_first)
    plot_graph(n_lst, theta_l, connected_comp_sizes)


def plot_histo(data, n):
    categories = list(data.keys())
    counts = [data[key] for key in categories]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color='skyblue')
    plt.xlabel(f'biggest component sizes for n={n}')
    plt.ylabel('Count')
    plt.title(f'Histogram Of Biggest Components Sizes for 30 Random Graphs Of Size {n}')
    plt.xticks(rotation=45)
    plt.show()


def plot_graph(n_lst: List[int], theta_l, connected_comp_sizes):

    # Create a figure with logarithmic x-axis
    plt.figure(figsize=(10, 6))
    plt.xscale('log')  # Apply logarithmic scale

    # Plot theta_l vs n with logarithmic x-axis
    for index, theta in enumerate(theta_l):
        plt.plot(n_lst, theta, label=f'theta_l{index+1} vs n', marker='o', linestyle='--')
        theta = [3 * x for x in theta]
        plt.plot(n_lst, theta, label=f'3 * theta_l{index+1} vs n', marker='o', linestyle='--')

    # Plot each sublist against n_lst with logarithmic x-axis
    markers = ['x', 's', '^', 'p', '*', '+']  # Example marker styles
    for index, sublist in enumerate(connected_comp_sizes):
        if index % 5 == 0:  # Selective plotting
            marker_style = markers[index % len(markers)]  # Cycle through marker styles
            if index == 0:
                plt.plot(n_lst, sublist, label=f'{index + 1} biggest connected comp size', marker=marker_style)
            else:
                plt.plot(n_lst, sublist, label=f'{index} biggest connected comp size', marker=marker_style)

    plt.xlabel('n')
    plt.ylabel('Sizes of connected components')
    plt.title('n vs Connected Components Sizes for Different k')
    plt.legend()
    plt.grid(True, which="both", ls="-")  # Enhanced grid visibility for log scale
    plt.show()


def connected_components_sizes(p: float, n: int):
    """
    :param p: probability for an edge
    :param n: number of vertices
    :return: histogram of connected components sizes (sorted)
    """
    random_graph = nx.fast_gnp_random_graph(n, p)
    # Find connected components
    components = list(nx.connected_components(random_graph))

    # Sort the components by size in descending order
    sorted_components = sorted(components, key=len, reverse=True)
    sizes_components = [len(i) for i in sorted_components]
    histo_components = Counter(sizes_components)
    sorted_histo = Counter({k: v for k, v in sorted(histo_components.items(), key=lambda item: item[1], reverse=True)})
    return sorted_histo


''' 
Very Subcritical:
 p = 1/2*n
 L1 ~ theta(ln(n))
 Lk ~ L1 for every fixed k
 for different sizes of n
'''


def very_subcritical_p_theta(n_lst: List[int]):
    """
    :param n_lst: all n values
    :return: the p values of very subcritical case (1 / (2 * n_val))
    """
    p_lst = [(1 / (2 * n_val)) for n_val in n_lst]
    theta_lst = [math.log(n, math.e) for n in n_lst]
    return p_lst, [theta_lst]


''' 
Barely Subcritical:
 p = (1 — eps)/n = 1/n - n^(0.01) / n^(4/3)
 eps = lambda / n^(1/3) = n^(0.01) / n^(1/3)
 lambda = n^(0.01)
 L1 ~ theta(ln(lambda)/ (eps^2))= 0.01ln(n) * n^(2/3)/ n^(0.02)
 Lk ~ L1 for every fixed k
 for different sizes of n
'''


def barely_subcritical_p_theta(n_lst: List[int]):
    """
    :param n_lst: all n values
    :return: the p values of barely subcritical case
    """
    p_lst = [(1 / n - (n ** 0.01 / n ** (4 / 3))) for n in n_lst]
    theta_lst = [((0.01 * math.log(n, math.e) * n**(2/3)) / n**0.02) for n in n_lst]
    return p_lst, [theta_lst]


''' 
The Critical Window:
 p = 1/n ± 2n^(-4/3)
 L1 ~ theta(n^(2/3))
 Lk ~ L1 for every fixed k
 for different sizes of n
'''


def critical_window_p_theta(n_lst: List[int]):
    """
    :param n_lst: all n values
    :return: the p values of The Critical Window case
    """
    p_lst = [(1 / n + (2 * n ** (-4 / 3))) for n in n_lst]
    theta_lst = [(n ** (2/3)) for n in n_lst]
    return p_lst, [theta_lst]


''' 
Barely Supercritical:
 p = (1 + eps)/n = 1/n + n^(0.01) / n^(4/3)
 eps = lambda / n^(1/3) = n^(0.01) / n^(1/3)
 lambda = n^(0.01)
 L1 ~ theta(2 * lambda * n^(2/3))= 2 * n^(0.01) * n^(2/3)
 The largest component has complexity approaching infinity
 All other components are simple
 L2 ~ ln(lambda) * eps^(-2) = 0.01*ln(n) * (n^(2/3)/n^0.02)
'''


def barely_sup_p_theta(n_lst: List[int]):
    """
    :param n_lst: all n values
    :return: the p values of The Critical Window case
    """
    p_lst = [(1/n + n**0.01 / n**(4/3)) for n in n_lst]
    l1_lst = [(2 * n**0.01 * n**(2/3)) for n in n_lst]
    l2_lst = [(0.01 * math.log(n, math.e) * (n**(2/3) / n**0.02)) for n in n_lst]
    return p_lst, [l1_lst, l2_lst]


''' 
Very Supercritical:
 p = c / n
 c > 1
 L1 ~ yn, 1-y = e^-yc
 c = 2, y = 0.796
 L2 ~ ln(n) 
'''


def very_super_p_theta(n_lst: List[int]):
    """
    :param n_lst: all n values
    :return: the p values of The Very Supercritical
    """
    p_lst = [(2/n) for n in n_lst]
    l1_lst = [(0.796 * n) for n in n_lst]
    l2_lst = [(math.log(n, math.e)) for n in n_lst]
    return p_lst, [l1_lst, l2_lst]


''' 
Very Supercritical Second Assign:
 p = c / n
 c > 1
 L1 ~ yn, 1-y = e^-yc
 c = 5, y = 0.993
 L2 ~ ln(n) 
'''


def very_super_p_theta_2(n_lst: List[int]):
    """
    :param n_lst: all n values
    :return: the p values of The Very Supercritical
    """
    p_lst = [(5/n) for n in n_lst]
    l1_lst = [(0.993 * n) for n in n_lst]
    l2_lst = [(math.log(n, math.e)) for n in n_lst]
    return p_lst, [l1_lst, l2_lst]


# Define the range of n values you want to plot
n_values = [10, 100, 1000, 10000, 100000, 1000000]  # Adjust this list as needed
cases = [very_subcritical_p_theta, barely_subcritical_p_theta, critical_window_p_theta, barely_sup_p_theta, very_super_p_theta, very_super_p_theta_2]

for case in cases:
    p_values, theta_values = case(n_values)
    calculate_k_connected_components_for_all_n(n_values, p_values, theta_values)
