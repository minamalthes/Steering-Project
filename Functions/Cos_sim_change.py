import torch
import torch.nn.functional as F
from Embeddings import import_embedding_data_from_pkl
from Steering_vector import import_steering_vector_from_pkl
from Steering import get_steered_embeddings
import plotly.graph_objects as go
import numpy as np

def calculate_cosine_similarities(steered_embeddings, steering_vector, show_all=False, original_embeddings=None):
    """
    Compute the cosine similarity between the steered embeddings and the steering vector, individually for each text.
    """
    if original_embeddings is not None:
        print("Original ranking: ")
        print("------------------")
        calculate_cosine_similarities(original_embeddings, steering_vector, show_all=False, original_embeddings=None)
        print("\nSteered ranking: ")
        print("------------------")

    cos_sims = []
    for i in range(len(steered_embeddings)):
        cos_sim = F.cosine_similarity(steered_embeddings[i].unsqueeze(0), steering_vector, dim=1)
        cos_sims.append(cos_sim.item())

    cos_sims_tensor = torch.tensor(cos_sims)

    top_10_indices = torch.topk(cos_sims_tensor, 10).indices

    print("\nTop 10 texts with highest cosine similarity to steering vector:\n")
    print(f"{'Text':<5} {'Title':<40} {'Genre':<15} {'Cosine Sim':>10}")
    print('-'*75)
    for idx in top_10_indices:
        i = idx.item()
        title = all_texts_data['title'][i][:37] + '...' if len(all_texts_data['title'][i]) > 40 else all_texts_data['title'][i]
        genre = all_texts_data['genre'][i]
        cos_val = cos_sims_tensor[i].item()
        print(f"{i+1:<5} {title:<40} {genre:<15} {cos_val:>10.4f}")


    if show_all:
        print(f"{'Text':<5} {'Title':<40} {'Genre':<15} {'Cosine Sim':>10}")
        print('-'*75)
        for i, cos_val in enumerate(cos_sims):
            title = all_texts_data['title'][i][:37] + '...' if len(all_texts_data['title'][i]) > 40 else all_texts_data['title'][i]
            genre = all_texts_data['genre'][i]
            print(f"{i+1:<5} {title:<40} {genre:<15} {cos_val:>10.4f}")

    #Print the top result title and description
    print("\nTop result: ")
    print("Title: " + all_texts_data['title'][top_10_indices[0].item()])
    print("Description: " + all_texts_data['overview'][top_10_indices[0].item()])

    return top_10_indices, cos_sims




def rank_categories_by_cosine_similarity(cos_sims, print_results=True, plot_bar_graph=True, remove_low_count=True, original_cos_sims=None):
    """
    Rank categories based on mean cosine similarity to steering vector
    """
    # Create a dictionary to store cosine similarities for each genre
    genre_cosines = {}
    
    for i, cos_sim in enumerate(cos_sims):
        genre = all_texts_data['genre'][i]
        if genre not in genre_cosines:
            genre_cosines[genre] = []
        genre_cosines[genre].append(cos_sim)
    
    # If original cosine similarities are provided, organize them by genre too
    original_genre_cosines = {}
    
    if original_cos_sims is not None:
        for i, cos_sim in enumerate(original_cos_sims):
            genre = all_texts_data['genre'][i]
            if genre not in original_genre_cosines:
                original_genre_cosines[genre] = []
            original_genre_cosines[genre].append(cos_sim)
    
    # Calculate mean cosine similarity for each genre
    genre_means = {}
    for genre, sims in genre_cosines.items():
        genre_means[genre] = sum(sims) / len(sims)
    
    # Sort genres by mean cosine similarity (steered)
    sorted_genres = sorted(genre_means.items(), key=lambda x: x[1], reverse=True)
    
    # Filter out genres with less than 25 counts
    if remove_low_count:
        filtered_genres = [(genre, mean_sim) for genre, mean_sim in sorted_genres 
                           if len(genre_cosines[genre]) >= 25]
    else:
        # If not removing low counts, keep all genres
        filtered_genres = sorted_genres

    filter_text = " (25+ Movies)" if remove_low_count else " (All Genres)"

    if print_results:
        # Also print the original cosine similarities if available
        if original_cos_sims is not None:
            # Calculate original genre means and sort them properly
            original_genre_means = {}
            for genre, sims in original_genre_cosines.items():
                original_genre_means[genre] = sum(sims) / len(sims)
            
            # Sort original genres by their original mean cosine similarity (highest to lowest)
            sorted_original_genres = sorted(original_genre_means.items(), key=lambda x: x[1], reverse=True)
            
            # Apply the same filtering as for steered genres
            if remove_low_count:
                filtered_original_genres = [(genre, mean_sim) for genre, mean_sim in sorted_original_genres 
                                           if len(original_genre_cosines[genre]) >= 25]
            else:
                filtered_original_genres = sorted_original_genres
            
            print(f"\nOriginal Category Ranking by Mean Cosine Similarity{filter_text}:")
            print("=" * 50)
            print(f"{'Genre':<20} {'Mean Cos Sim':<15} {'Count':<10}")
            print("-" * 50)
            for genre, mean_sim in filtered_original_genres:
                count = len(original_genre_cosines[genre])
                print(f"{genre:<20} {mean_sim:<15.4f} {count:<10}")

        # Print the steered category ranking
        print(f"\nSteered Category Ranking by Mean Cosine Similarity{filter_text}:")
        print("=" * 50)
        print(f"{'Genre':<20} {'Mean Cos Sim':<15} {'Count':<10}")
        print("-" * 50)
        for genre, mean_sim in filtered_genres:
            count = len(genre_cosines[genre])
            print(f"{genre:<20} {mean_sim:<15.4f} {count:<10}")


    
    if plot_bar_graph:
        plot_bar_category_ranking(sorted_genres, genre_cosines, remove_low_count=remove_low_count, original_genre_cosines=original_genre_cosines)
    
    return sorted_genres, genre_means




def plot_bar_category_ranking(sorted_genres, genre_cosines, remove_low_count=True, original_genre_cosines=None):
    """
    Create a column graph showing category rankings with positive values,
    coloring negative original values in red, with original values as grey overlay
    """
    # Filter out genres with less than 25 counts
    if remove_low_count:
        filtered_genres = [(genre, mean_sim) for genre, mean_sim in sorted_genres 
                           if len(genre_cosines[genre]) >= 25]
    else:
        # If not removing low counts, keep all genres
        filtered_genres = sorted_genres
    
    if not filtered_genres:
        print("No genres have 25 or more movies. Skipping plot.")
        return None
    
    genres = [genre for genre, _ in filtered_genres]
    mean_sims = [mean_sim for _, mean_sim in filtered_genres]
    counts = [len(genre_cosines[genre]) for genre, _ in filtered_genres]
    
    # Create colors: red for originally negative values, blue for positive
    colors = ['red' if mean_sim < 0 else 'blue' for _, mean_sim in filtered_genres]
    
    # Convert to absolute values for display
    #abs_mean_sims = [abs(sim) for sim in mean_sims]
    
    # Create hover text with steered values and counts
    hover_text_steered = [f"Genre: {genre}<br>Steered Mean Cosine Similarity: {mean_sim:.4f}<br>Count: {count}" 
                         for genre, mean_sim, count in zip(genres, mean_sims, counts)]
    
    # Create the figure
    fig = go.Figure()
    
    # Add original cosine similarities as grey overlay if provided
    if original_genre_cosines is not None:
        # Calculate original mean cosine similarities for filtered genres
        original_mean_sims = []
        hover_text_original = []
        
        for genre, _ in filtered_genres:
            if genre in original_genre_cosines:
                original_mean = sum(original_genre_cosines[genre]) / len(original_genre_cosines[genre])
                original_mean_sims.append(original_mean) # Can also do abs(original_mean) if you want to show absolute values
                hover_text_original.append(f"Genre: {genre}<br>Original Mean Cosine Similarity: {original_mean:.4f}<br>Count: {len(original_genre_cosines[genre])}")
            else:
                original_mean_sims.append(0)
                hover_text_original.append(f"Genre: {genre}<br>Original Mean Cosine Similarity: N/A<br>Count: 0")
    
    # Add the steered embeddings bars (main bars)
    fig.add_trace(go.Bar(
        x=genres,
        y=mean_sims,
        marker_color=colors,
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_text_steered,
        name='Steered',
        opacity=1
    ))

    # Add the original bars (grey overlay)
    fig.add_trace(go.Bar(
        x=genres,
        y=original_mean_sims,
        marker_color='grey',
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_text_original,
        name='Original',
        opacity=1
    ))
    
    if remove_low_count:
        fig.update_layout(title=f'Category Ranking by Mean Cosine Similarity (25+ Movies) | Feature: {feature}')
    else:
        fig.update_layout(title=f'Category Ranking by Mean Cosine Similarity (All Genres) | Feature: {feature}')
    fig.update_layout(
        xaxis_title='Genre',
        yaxis_title='Absolute Mean Cosine Similarity',
        xaxis_tickangle=-45,
        width=900,
        height=500,
        showlegend=True,
        barmode='overlay'  # This makes the bars overlay instead of grouping
    )
    
    fig.show()
    
    return fig


def calculate_original_cosine_similarities(original_embeddings, steering_vector):
    """
    Calculate cosine similarities between original embeddings and steering vector
    """
    original_cos_sims = []
    for i in range(len(original_embeddings)):
        cos_sim = F.cosine_similarity(original_embeddings[i].unsqueeze(0), steering_vector, dim=1)
        original_cos_sims.append(cos_sim.item())
    return original_cos_sims


def get_most_shifted_categories(steered_cos_sims, original_cos_sims, remove_low_count=True, top_n=10, print_results=True):
    """
    Calculate and print the categories with the largest change in cosine similarity (most shifted)
    """
    # Create dictionaries to store cosine similarities for each genre
    steered_genre_cosines = {}
    original_genre_cosines = {}
    
    # Organize cosine similarities by genre
    for i, (steered_sim, original_sim) in enumerate(zip(steered_cos_sims, original_cos_sims)):
        genre = all_texts_data['genre'][i]
        
        if genre not in steered_genre_cosines:
            steered_genre_cosines[genre] = []
            original_genre_cosines[genre] = []
        
        steered_genre_cosines[genre].append(steered_sim)
        original_genre_cosines[genre].append(original_sim)
    
    # Calculate mean cosine similarities and shifts for each genre
    genre_shifts = {}
    for genre in steered_genre_cosines.keys():
        steered_mean = sum(steered_genre_cosines[genre]) / len(steered_genre_cosines[genre])
        original_mean = sum(original_genre_cosines[genre]) / len(original_genre_cosines[genre])
        shift = steered_mean - original_mean
        count = len(steered_genre_cosines[genre])
        
        genre_shifts[genre] = {
            'shift': shift,
            'steered_mean': steered_mean,
            'original_mean': original_mean,
            'count': count
        }
    
    # Filter by count if requested
    if remove_low_count:
        genre_shifts = {k: v for k, v in genre_shifts.items() if v['count'] >= 25}
    
    if not genre_shifts:
        print("No genres have 25 or more movies. Skipping analysis.")
        return None
    
    # Sort by absolute shift (largest changes first)
    sorted_shifts = sorted(genre_shifts.items(), key=lambda x: x[1]['shift'], reverse=True)
    
    # Take top N most shifted
    top_shifts = sorted_shifts[:top_n]
    
    # Print summary
    if print_results:
        filter_text = " (25+ Movies)" if remove_low_count else " (All Genres)"
        print(f"\nTop {top_n} Most Shifted Categories{filter_text}:")
        print("=" * 70)
        print(f"{'Genre':<20} {'Shift':<10} {'Original':<10} {'Steered':<10} {'Count':<8}")
        print("-" * 70)
        for genre, data in top_shifts:
            print(f"{genre:<20} {data['shift']:<10.4f} {data['original_mean']:<10.4f} {data['steered_mean']:<10.4f} {data['count']:<8}")
    
    return top_shifts

def get_top_shift_for_steering_coefficient(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector):
    """
    Get the shift data for all categories at a specific steering coefficient
    """
    steered_embeddings = get_steered_embeddings(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector, normalize=True)
    top_10_indices, all_cos_sims = calculate_cosine_similarities(steered_embeddings, steering_vector, show_all=False, original_embeddings=None)
    top_shifts = get_most_shifted_categories(all_cos_sims, original_cos_sims, remove_low_count=True, top_n=10, print_results=False)
    
    # Convert to dictionary for easier access
    shift_dict = {}
    for genre, data in top_shifts:
        shift_dict[genre] = data['shift']
    
    return shift_dict
    
def plot_change_in_top_shift(model, encoded_input, layer_to_steer, steering_vector, steering_coefficient_start, steering_coefficient_end, number_of_steps=5):
    """
    Plot how the top shifted categories change across different steering coefficients
    """
    steering_coefficients = torch.linspace(steering_coefficient_start, steering_coefficient_end, number_of_steps)
    
    # Dictionary to store shifts for each category across all steering coefficients
    category_shifts = {}
    
    print(f"Analyzing shifts across {number_of_steps} steering coefficients from {steering_coefficient_start} to {steering_coefficient_end}")
    
    for i, steering_coefficient in enumerate(steering_coefficients):
        print(f"\nProcessing steering coefficient {i+1}/{number_of_steps}: {steering_coefficient:.3f}")
        
        # Get shifts for this steering coefficient
        shift_dict = get_top_shift_for_steering_coefficient(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector)
        
        # Store results for each category
        for category, shift in shift_dict.items():
            if category not in category_shifts:
                category_shifts[category] = []
            category_shifts[category].append(shift)
    
    # Create the plot
    fig = go.Figure()
    
    # Sort categories by their final shift value (highest shift first)
    categories_with_final_shifts = [(cat, shifts[-1]) for cat, shifts in category_shifts.items()]
    categories_with_final_shifts.sort(key=lambda x: x[1], reverse=True)
    
    # Add a line for each category in sorted order
    for category, _ in categories_with_final_shifts:
        shifts = category_shifts[category]
        fig.add_trace(go.Scatter(
            x=steering_coefficients.numpy(),
            y=shifts,
            mode='lines+markers',
            name=category,
            line=dict(width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title=f'Change in Genre Shift vs Steering Coefficient | Layer: {layer_to_steer} | Feature: {feature}',
        xaxis_title='Steering Coefficient',
        yaxis_title='Shift in Mean Cosine Similarity',
        width=900,
        height=600,
        showlegend=True
    )
    
    fig.show()
    
    return category_shifts, steering_coefficients.numpy()

def plot_rank_plot(category_shifts, steering_coefficients):
    """
    Plot rank changes of categories over steering coefficients.
    """
    categories = list(category_shifts.keys())
    num_steps = len(steering_coefficients)
    
    # Build a rank matrix: rows = categories, cols = coefficients
    ranks_over_coeffs = []
    for step in range(num_steps):
        # Get the shift values at this coefficient for all categories
        shifts_at_step = [category_shifts[cat][step] for cat in categories]
        # Get sorted indices (highest shift gets rank 1)
        sorted_idx = np.argsort(shifts_at_step)[::-1]
        # Map indices to ranks
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, len(categories)+1)
        ranks_over_coeffs.append(ranks)
    
    # Transpose to get list of ranks for each category
    ranks_over_coeffs = np.array(ranks_over_coeffs).T  # shape: (num_categories, num_steps)
    
    # Get final ranking (at last steering coefficient) to sort legend
    final_ranks = ranks_over_coeffs[:, -1]  # Last column contains final ranks
    # Sort categories by their final rank (rank 1 first)
    sorted_category_indices = np.argsort(final_ranks)
    
    # Plot
    fig = go.Figure()
    for idx in sorted_category_indices:
        cat = categories[idx]
        fig.add_trace(go.Scatter(
            x=steering_coefficients,
            y=ranks_over_coeffs[idx],
            mode='lines+markers',
            name=cat,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f'Rank Plot: Category rank vs Steering Coefficient | Layer: {layer_to_steer} | Feature: {feature}',
        xaxis_title='Steering Coefficient',
        yaxis_title='Rank (1=highest shift)',
        yaxis_autorange='reversed',  # so rank 1 is at the top
        width=900,
        height=600,
        showlegend=True
    )
    fig.show()

'''
# Example usage
# Import data
data = import_embedding_data_from_pkl('Test_export_embeddings.pkl', model=True, embeddings=True, encoded_input=True, all_texts_data=True)
model, original_embeddings, encoded_input, all_texts_data = data

feature = "Love"
layer_to_steer = 11  # Change this to the layer you want to analyze
steering_coefficient = 0.5  # Adjust steering coefficient as needed

steering_vector = import_steering_vector_from_pkl('steering_vector.pkl', feature_name=feature, layer_to_steer=layer_to_steer)
steered_embeddings = get_steered_embeddings(model, encoded_input, layer_to_steer=layer_to_steer, steering_coefficient=steering_coefficient, steering_vector=steering_vector, normalize=True)

print(f"\nSteering | Feature: {feature} | Coefficient: {steering_coefficient} | Layer: {layer_to_steer}\n")

# Calculate cosine similarities and rankings
original_cos_sims = calculate_original_cosine_similarities(original_embeddings, steering_vector)
top_10_indices, cos_sims = calculate_cosine_similarities(steered_embeddings, steering_vector, show_all=False, original_embeddings=original_embeddings)
sorted_genres, genre_means = rank_categories_by_cosine_similarity(cos_sims, print_results=True, plot_bar_graph=True, 
                                                                  remove_low_count=True, original_cos_sims=original_cos_sims)

top_shifts = get_most_shifted_categories(cos_sims, original_cos_sims, remove_low_count=True, top_n=10, print_results=True)

category_shifts, steering_coefficients = plot_change_in_top_shift(model, encoded_input, layer_to_steer, steering_vector, 
                                                                  steering_coefficient_start=0.0, steering_coefficient_end=0.5, 
                                                                  number_of_steps=5)
plot_rank_plot(category_shifts, steering_coefficients)
'''