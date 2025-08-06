from Steering import get_steered_embeddings_neuron, get_steered_embeddings_vector
from Steering_vector import get_steering_vector, import_feature_texts, import_steering_vector_from_pkl
from Embeddings import set_model_and_tokenizer, import_embedding_data_from_pkl

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np


def convergence_check_with_vector(model, encoded_input, layer_to_steer, steering_vector, feature, start_coeff=0, end_coeff=100, normalize=True, mode='l2', 
                      derivative=False, step_size=1, grid=True, text_range=None):
    """
    Plot the distance between steered embeddings and the steering vector over a range of steering coefficients. 
    
    Args:
        model: The transformer model.
        encoded_input (dict): Tokenized input data.
        layer_to_steer (int): Layer index to apply steering to.
        steering_vector (torch.Tensor): Steering vector to use.
        feature (str): Feature name for plot title.
        start_coeff (int, optional): Starting steering coefficient. Defaults to 0.
        end_coeff (int, optional): Ending steering coefficient. Defaults to 100.
        normalize (bool, optional): Whether to normalize embeddings. Defaults to True.
        mode (str, optional): Distance metric ('l2', 'l1', 'cosine'). Defaults to 'l2'.
        derivative (bool, optional): Whether to plot derivative. Defaults to False.
        step_size (int, optional): Step size between coefficients. Defaults to 1.
        grid (bool, optional): Whether to show grid on plot. Defaults to True.
        text_range (tuple, optional): Range of texts to analyze. Defaults to None.
    
    Note:
        Uses steering vector for steering and finds potential convergence points.

    """

    steering_coefficients = np.arange(start_coeff, end_coeff + step_size, step_size).tolist()

    diffs = []

    if text_range is not None:
        encoded_input = {
            key: val[slice(*text_range)] for key, val in encoded_input.items()
        }
    
    print("Processing convergence check...")
    total = len(steering_coefficients)
    for i, coeff in enumerate(steering_coefficients):
        if i % (total // 10) == 0:
            percent = round((i + 1) / total * 100)
            print(f"{percent}% done", end='\r')
        steered_embeddings = get_steered_embeddings_vector(model, encoded_input, layer_to_steer, coeff, steering_vector, normalize=normalize, verbose=False)
        if mode == 'l2':
            diff = torch.norm(steered_embeddings - steering_vector, dim=1).mean().item()
        elif mode == 'l1':
            diff = torch.abs(steered_embeddings - steering_vector).sum(dim=1).mean().item()
        elif mode == 'cosine':
            similarity = F.cosine_similarity(steered_embeddings, steering_vector, dim=1)
            diff = (1 - similarity).mean().item()  # smaller = more aligned

        diffs.append(diff)

    # Plotting
    plt.plot(steering_coefficients, diffs, "r-")
    plt.xlabel("Steering Coefficient")
    ylabel = {'l2': f"Mean L2 Distance", 
              'l1': f"Mean L1 Distance", 
              'cosine': f"Mean (1 - Cosine Similarity)"}.get(mode, "Distance to Steering Vector")
    plt.ylabel(ylabel)
    plt.title(f"Convergence: {ylabel} | Layer {layer_to_steer} | Feature: {feature}")
    plt.grid(grid)
    plt.show()

    # Print distance at 0
    if 0 in steering_coefficients:
        zero_index = steering_coefficients.index(0)
        zero_distance = diffs[zero_index]
        print(f"Distance at coefficient 0: {zero_distance:.4f}")

    # Print min and max distances
    min_diff = min(diffs)
    min_index = diffs.index(min_diff)
    min_coeff = steering_coefficients[min_index]
    print(f"Minimum distance: {min_diff:.4f} at coefficient ≈ {min_coeff:.3f}")

    max_diff = max(diffs)
    max_index = diffs.index(max_diff)
    max_coeff = steering_coefficients[max_index]
    print(f"Maximum distance: {max_diff:.4f} at coefficient ≈ {max_coeff:.3f}")

    slope_threshold = 0.001
    patience = 3  # Number of consecutive steps to check for convergence

    # Split index: where non-negative coefficients start
    split_index = next((i for i, c in enumerate(steering_coefficients) if c >= 0), len(steering_coefficients))

    def find_convergence(start, end, label, reverse=False):
        indices = list(range(start, end - patience))
        if reverse:
            indices = indices[::-1]  # From 0 → more negative
        else:
            indices = indices  # From 0 → more positive
        stable_start = None
        current_region = []
        for i in indices:
            window = diffs[i:i + patience + 1]
            if len(window) < patience + 1:
                continue
            deltas = [abs(window[j + 1] - window[j]) for j in range(patience)]
            if all(d < slope_threshold for d in deltas):
                if not current_region:
                    stable_start = i  # New stable region starts
                current_region.append(i)
            else:
                if current_region:
                    # Reset tracking if streak is broken
                    current_region = []
        if stable_start is not None:
            coeff = steering_coefficients[stable_start + patience]
            value = diffs[stable_start + patience]
            print(f"Converged ({label}) around coefficient ≈ {coeff:.3f} | Distance: {value:.4f}")


    # Negative side (reversed to start from 0 → more negative)
    if split_index > patience:
        find_convergence(0, split_index, "negative side", reverse=True)

    # Positive side (normal direction)
    if len(diffs) - split_index > patience:
        find_convergence(split_index, len(diffs), "positive side", reverse=False)
    
    print(f"Final distance: {diffs[-1]:.4f} at coefficient = {steering_coefficients[-1]:.3f}")

    # If requested, plot the derivative of the distance
    if derivative:
        plot_derivative(steering_coefficients, diffs, ylabel, layer_to_steer, feature, neuron=None, grid=grid)


def convergence_check_with_neuron(model, encoded_input, neuron, layer_to_steer, comparison_vector, feature, start_coeff=0, end_coeff=100, normalize=True, mode='l2', 
                                  derivative=False, step_size=1, grid=True, text_range=None):
    """
    Plot the distance between steered embeddings and a comparison vector over a range of steering coefficients.
    
    Args:
        model: The transformer model.
        encoded_input (dict): Tokenized input data.
        neuron (int): Single neuron index to use for steering.
        layer_to_steer (int): Layer index to apply steering to.
        comparison_vector (torch.Tensor): Vector to compare against.
        feature (str): Feature name for plot title.
        start_coeff (int, optional): Starting steering coefficient. Defaults to 0.
        end_coeff (int, optional): Ending steering coefficient. Defaults to 100.
        normalize (bool, optional): Whether to normalize embeddings. Defaults to True.
        mode (str, optional): Distance metric ('l2', 'l1', 'cosine'). Defaults to 'l2'.
        derivative (bool, optional): Whether to plot derivative. Defaults to False.
        step_size (int, optional): Step size between coefficients. Defaults to 1.
        grid (bool, optional): Whether to show grid on plot. Defaults to True.
        text_range (tuple, optional): Range of texts to analyze. Defaults to None.
    
    Note:
        Uses single neuron steering and finds potential convergence points.
    """

    steering_coefficients = np.arange(start_coeff, end_coeff + step_size, step_size).tolist()

    diffs = []

    if text_range is not None:
        encoded_input = {
            key: val[slice(*text_range)] for key, val in encoded_input.items()
        }
    
    print("Processing convergence check...")
    total = len(steering_coefficients)
    for i, coeff in enumerate(steering_coefficients):
        if i % (total // 10) == 0:
            percent = round((i + 1) / total * 100)
            print(f"{percent}% done", end='\r')
        steered_embeddings = get_steered_embeddings_neuron(model, encoded_input, layer_to_steer, neuron, coeff, normalize=normalize, verbose=False)
        if mode == 'l2':
            diff = torch.norm(steered_embeddings - comparison_vector, dim=1).mean().item()
        elif mode == 'l1':
            diff = torch.abs(steered_embeddings - comparison_vector).sum(dim=1).mean().item()
        elif mode == 'cosine':
            similarity = F.cosine_similarity(steered_embeddings, comparison_vector, dim=1)
            diff = (1 - similarity).mean().item()  # smaller = more aligned

        diffs.append(diff)

    # Plotting
    plt.plot(steering_coefficients, diffs, "r-")
    plt.xlabel("Steering Coefficient")
    ylabel = {'l2': f"Mean L2 Distance", 
              'l1': f"Mean L1 Distance", 
              'cosine': f"Mean (1 - Cosine Similarity)"}.get(mode, "Distance to Steering Vector")
    plt.ylabel(ylabel)
    plt.title(f"Convergence: {ylabel} | Neuron {neuron} | Layer {layer_to_steer} | Comparison feature: {feature}")
    plt.grid(grid)
    plt.show()

    # Print distance at 0
    if 0 in steering_coefficients:
        zero_index = steering_coefficients.index(0)
        zero_distance = diffs[zero_index]
        print(f"Distance at coefficient 0: {zero_distance:.4f}")

    # Print min and max distances
    min_diff = min(diffs)
    min_index = diffs.index(min_diff)
    min_coeff = steering_coefficients[min_index]
    print(f"Minimum distance: {min_diff:.4f} at coefficient ≈ {min_coeff:.3f}")

    max_diff = max(diffs)
    max_index = diffs.index(max_diff)
    max_coeff = steering_coefficients[max_index]
    print(f"Maximum distance: {max_diff:.4f} at coefficient ≈ {max_coeff:.3f}")

    slope_threshold = 0.001
    patience = 3  # Number of consecutive steps to check for convergence

    # Split index: where non-negative coefficients start
    split_index = next((i for i, c in enumerate(steering_coefficients) if c >= 0), len(steering_coefficients))

    def find_convergence(start, end, label, reverse=False):
        indices = list(range(start, end - patience))
        if reverse:
            indices = indices[::-1]  # From 0 → more negative
        else:
            indices = indices  # From 0 → more positive
        stable_start = None
        current_region = []
        for i in indices:
            window = diffs[i:i + patience + 1]
            if len(window) < patience + 1:
                continue
            deltas = [abs(window[j + 1] - window[j]) for j in range(patience)]
            if all(d < slope_threshold for d in deltas):
                if not current_region:
                    stable_start = i  # New stable region starts
                current_region.append(i)
            else:
                if current_region:
                    # Reset tracking if streak is broken
                    current_region = []
        if stable_start is not None:
            coeff = steering_coefficients[stable_start + patience]
            value = diffs[stable_start + patience]
            print(f"Converged ({label}) around coefficient ≈ {coeff:.3f} | Distance: {value:.4f}")


    # Negative side (reversed to start from 0 → more negative)
    if split_index > patience:
        find_convergence(0, split_index, "negative side", reverse=True)

    # Positive side (normal direction)
    if len(diffs) - split_index > patience:
        find_convergence(split_index, len(diffs), "positive side", reverse=False)
    
    print(f"Final distance: {diffs[-1]:.4f} at coefficient = {steering_coefficients[-1]:.3f}")

    # If requested, plot the derivative of the distance
    if derivative:
        plot_derivative(steering_coefficients, diffs, ylabel, layer_to_steer, feature, neuron, grid=grid)


def plot_derivative(steering_coefficients, diffs, ylabel, layer_to_steer, feature, neuron=None, grid=True):
    """
    Analyze and plot the derivative of distance/similarity values across a range of steering coefficients.
    
    Args:
        steering_coefficients (list): List of steering coefficient values.
        diffs (list): List of distance/similarity values.
        ylabel (str): Y-axis label for the plot.
        layer_to_steer (int): Layer index being analyzed.
        feature (str): Feature name for plot title.
        grid (bool, optional): Whether to show grid on plot. Defaults to True.
    
    Note:
        Identifies effective regions and convergence points based on derivative analysis.
    """

    derivatives = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]

    plt.plot(steering_coefficients[:-1], derivatives, "b-")
    plt.xlabel("Steering Coefficient")
    plt.ylabel("Derivative of Difference")
    if neuron is not None:
        plt.title(f"Derivative of {ylabel} | Neuron {neuron} | Layer {layer_to_steer} | Comparison feature: {feature}")
    else:
        plt.title(f"Derivative of {ylabel} | Layer {layer_to_steer} | Feature: {feature}")
    plt.grid(grid)
    plt.show()

    # "Effective region" span (where |derivative| > ε)
    epsilon = 0.002
    effective_indices = [i for i, d in enumerate(derivatives) if abs(d) > epsilon]
    if effective_indices:
        effective_start = effective_indices[0]
        effective_end = effective_indices[-1] + 1
        print(f"Effective region: coefficients ≈ [{steering_coefficients[effective_start]}, {steering_coefficients[effective_end]}]")

    else:
        effective_start = effective_end = None
        print("No effective region found (derivatives all below threshold).")

    # Total change across effective region
    if effective_start is not None:
        total_change = sum(abs(derivatives[i]) for i in range(effective_start, effective_end))
        print(f"Total change in effective region: {total_change:.4f}")
    else:
        total_change = 0

    # Time to 90% convergence
    cumulative_changes = []
    cumulative = 0
    for d in derivatives:
        cumulative += abs(d)
        cumulative_changes.append(cumulative)

    total_cumulative = cumulative_changes[-1] if cumulative_changes else 1
    threshold_90 = 0.9 * total_cumulative
    convergence_index = next((i for i, v in enumerate(cumulative_changes) if v >= threshold_90), None)
    if convergence_index is not None:
        print(f"90% of total change reached by coefficient ≈ {steering_coefficients[convergence_index]}")



def convergence_by_category(model, encoded_input, all_texts_data, layer_to_steer, steering_vector, feature, category_column='genre', start_coeff=0, end_coeff=100,
                            normalize=True, mode='l2', derivative=False, step_size=1, grid=True, text_range=None, min_count=5):
    """
    Plot the mean distance between steered embeddings and a steering vector for each category over a range of steering coefficients.
    
    Args:
        model: The transformer model.
        encoded_input (dict): Tokenized input data.
        all_texts_data (pd.DataFrame): Text metadata with category information.
        layer_to_steer (int): Layer index to apply steering to.
        steering_vector (torch.Tensor): Steering vector to use.
        feature (str): Feature name for plot title.
        category_column (str, optional): Column name for categorization. Defaults to 'genre'.
        start_coeff (int, optional): Starting steering coefficient. Defaults to 0.
        end_coeff (int, optional): Ending steering coefficient. Defaults to 100.
        normalize (bool, optional): Whether to normalize embeddings. Defaults to True.
        mode (str, optional): Distance metric ('l2', 'l1', 'cosine'). Defaults to 'l2'.
        derivative (bool, optional): Whether to plot derivative. Defaults to False.
        step_size (int, optional): Step size between coefficients. Defaults to 1.
        grid (bool, optional): Whether to show grid on plot. Defaults to True.
        text_range (tuple, optional): Range of texts to analyze. Defaults to None.
        min_count (int, optional): Minimum category count to include. Defaults to 5.
    
    Note:
        Analyzes convergence patterns separately for each category with sufficient data.
    """

    steering_coefficients = np.arange(start_coeff, end_coeff + step_size, step_size).tolist()

    # Apply slicing if requested
    if text_range is not None:
        s = slice(*text_range)
        encoded_input = {key: val[s] for key, val in encoded_input.items()}
        all_texts_data = all_texts_data.iloc[s]

    # Count how many times each category occurs
    category_counts = {}
    for i in range(len(all_texts_data)):
        category = all_texts_data.iloc[i][category_column]
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1

    # Filter categories with enough data
    valid_categories = []
    for cat, count in category_counts.items():
        if count >= min_count:
            valid_categories.append(cat)

    # Initialize empty lists for storing distances per category
    category_diffs = {}
    for cat in valid_categories:
        category_diffs[cat] = []

    print("Processing convergence by category...")
    total = len(steering_coefficients)
    for i, coeff in enumerate(steering_coefficients):
        if i % (total // 10) == 0:
            percent = round((i + 1) / total * 100)
            print(f"{percent}% done", end='\r')
        steered_embeddings = get_steered_embeddings_vector(model, encoded_input, layer_to_steer, coeff, steering_vector, normalize=normalize, verbose=False)
        if mode == 'l2':
            distances = torch.norm(steered_embeddings - steering_vector, dim=1).tolist()
        elif mode == 'l1':
            distances = torch.abs(steered_embeddings - steering_vector).sum(dim=1).tolist()
        elif mode == 'cosine':
            distances = (1 - F.cosine_similarity(steered_embeddings, steering_vector, dim=1)).tolist()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Collect distances for each category
        grouped = {}
        for cat in valid_categories:
            grouped[cat] = []

        for i, dist in enumerate(distances):
            cat = all_texts_data.iloc[i][category_column]
            if cat in grouped:
                grouped[cat].append(dist)

        # Compute and store mean distance per category for this coefficient
        for cat in valid_categories:
            dists = grouped[cat]
            if len(dists) > 0:
                mean_diff = sum(dists) / len(dists)
            else:
                mean_diff = None  # category has no data at this step
            category_diffs[cat].append(mean_diff)

    # Plotting
    for cat in valid_categories:
        values = category_diffs[cat]
        plt.plot(steering_coefficients, values, label=cat)

    plt.xlabel("Steering Coefficient")
    ylabel = {'l2': f"Mean L2",
              'l1': f"Mean L1",
              'cosine': f"Mean (1 - Cosine Similarity)"}.get(mode, "Distance to Steering Vector")
    plt.ylabel(ylabel)
    plt.title(f"Convergence by {category_column.capitalize()}: {ylabel} | Layer {layer_to_steer} | Feature: {feature}")
    plt.grid(grid)
    plt.legend()
    plt.show()

    # Optional: Slope-based convergence detection
    print("\nConvergence analysis by category (slope-based):")
    print("=" * 50)

    slope_threshold = 0.001
    patience = 3  # Number of consecutive small steps required for convergence

    split_index = next((i for i, c in enumerate(steering_coefficients) if c >= 0), len(steering_coefficients))

    def find_convergence(diffs, label, start, end, reverse=False):
        indices = list(range(start, end - patience))
        if reverse:
            indices = indices[::-1]
        stable_start = None
        current_region = []
        for i in indices:
            window = diffs[i:i + patience + 1]
            if len(window) < patience + 1 or any(x is None for x in window):
                continue
            deltas = [abs(window[j + 1] - window[j]) for j in range(patience)]
            if all(d < slope_threshold for d in deltas):
                if not current_region:
                    stable_start = i
                current_region.append(i)
            else:
                current_region = []
        if stable_start is not None:
            coeff = steering_coefficients[stable_start + patience]
            value = diffs[stable_start + patience]
            print(f" Converged ({label}) around coefficient ≈ {coeff} | Distance: {value:.4f}")

    for cat in valid_categories:
        diffs = category_diffs[cat]
        print(f"\n{cat}:")
        valid_diffs = [d for d in diffs if d is not None]
        if not valid_diffs or len(valid_diffs) < patience + 1:
            print(" No valid data.")
            continue

        # min/max
        min_diff = min(valid_diffs)
        min_index = diffs.index(min_diff)
        print(f" Minimum distance: {min_diff:.4f} at coefficient ≈ {steering_coefficients[min_index]}")

        max_diff = max(valid_diffs)
        max_index = diffs.index(max_diff)
        print(f" Maximum distance: {max_diff:.4f} at coefficient ≈ {steering_coefficients[max_index]}")

        # convergence
        find_convergence(diffs, "negative side", 0, split_index, reverse=True)
        find_convergence(diffs, "positive side", split_index, len(diffs), reverse=False)


    if derivative:
        plt.figure(figsize=(10, 6))
        for cat, diffs in category_diffs.items():
            if len(diffs) < 2 or any(d is None for d in diffs):
                continue
            derivatives = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
            plt.plot(steering_coefficients[:-1], derivatives, label=cat)

        plt.xlabel("Steering Coefficient")
        plt.ylabel("Derivative of Difference")
        plt.title(f"Derivative of {ylabel} by Category | Layer {layer_to_steer} | Feature: {feature}")
        plt.grid(grid)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    #Example usage:
    # Import data
    data = import_embedding_data_from_pkl('Test_export_embeddings.pkl', model=True, embeddings=True, encoded_input=True, all_texts_data=True)
    model, original_embeddings, encoded_input, all_texts_data = data

    layer_to_steer = 11
    feature = "War"

    steering_vector = import_steering_vector_from_pkl('steering_vector.pkl', layer_to_steer=layer_to_steer, feature_name=feature)

    convergence_check_with_vector(model, 
                              encoded_input,
                              layer_to_steer, 
                              steering_vector, 
                              feature, 
                              start_coeff=-100, 
                              end_coeff=100, 
                              normalize=True,
                              mode='l2', 
                              derivative=True, 
                              step_size=1,
                              grid=True, 
                              text_range=(0,100))
    
    # Steer with neuron
    neuron = 69
    comparison_vector = steering_vector
    convergence_check_with_neuron(model, 
                              encoded_input,
                              neuron,
                              layer_to_steer, 
                              comparison_vector, 
                              feature, 
                              start_coeff=-100, 
                              end_coeff=100, 
                              normalize=True,
                              mode='l2', 
                              derivative=True, 
                              step_size=1,
                              grid=True, 
                              text_range=(0,100))
    
    convergence_by_category(model, 
                        encoded_input, 
                        all_texts_data, 
                        layer_to_steer, 
                        steering_vector, 
                        feature, 
                        category_column='genre', 
                        start_coeff=-100, 
                        end_coeff=100, 
                        normalize=True, 
                        mode='l2', 
                        derivative=True, 
                        step_size=1, 
                        grid=True, 
                        text_range=(0,100), 
                        min_count=5) 