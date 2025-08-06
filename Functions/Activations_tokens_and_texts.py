from Embeddings import set_model_and_tokenizer, get_embeddings, import_embedding_data_from_pkl
from Steering import create_steered_model_output
from Steering_vector import import_steering_vector_from_pkl
import matplotlib.pyplot as plt
import torch
import numpy as np

# Single token

def plot_activation_values_specific_token(hidden_states, layer, token_index, extreme_nodes=True):
    """
    Plot the activation values for a specific token in a specific layer.
    
    Args:
        hidden_states (tuple): Hidden states from model output.
        layer (int): Layer index to analyze.
        token_index (int): Index of the token to analyze.
        extreme_nodes (bool, optional): Whether to find extreme nodes. Defaults to True.
    """
    activation_values = hidden_states[layer][0, token_index, :].cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(activation_values)), activation_values)
    plt.xlabel('Neuron Index')
    plt.ylabel('Activation Value')
    plt.title(f'Activation Values for Token "{tokens[token_index]}" in Layer {layer}')
    plt.show()

    if extreme_nodes:
        find_extreme_nodes_for_specific_token(hidden_states, layer, token_index)


def find_extreme_nodes_for_specific_token(hidden_states, layer, token_index, Print = True):
    """
    Find the nodes with the least and most activation for a specific token in a specific layer.
    
    Args:
        hidden_states (tuple): Hidden states from model output.
        layer (int): Layer index to analyze.
        token_index (int): Index of the token to analyze.
        Print (bool, optional): Whether to print results. Defaults to True.
    
    Returns:
        tuple: (min_node, max_node) indices.
    """
    activation_values = hidden_states[layer][0, token_index, :].cpu().numpy()
    min_node = activation_values.argmin()
    max_node = activation_values.argmax()
    min_value = activation_values[min_node]
    max_value = activation_values[max_node]
    
    if Print:
        print(f"\nExtreme nodes for token \"{tokens[token_index]}\" in Layer {layer}:")
        print(f"{'Type':>10} {'Node':>8} {'Activation':>12}")
        print("-" * 32)
        print(f"{'Min':>10} {min_node:>8} {min_value:>12.4f}")
        print(f"{'Max':>10} {max_node:>8} {max_value:>12.4f}")
    
    return min_node, max_node

def plot_activation_values_all_layers(hidden_states, token_index, extreme_nodes=True):
    """
    Plot the activation values for a specific token across all layers.
    
    Args:
        hidden_states (tuple): Hidden states from model output.
        token_index (int): Index of the token to analyze.
        extreme_nodes (bool, optional): Whether to find extreme nodes. Defaults to True.
    """
    num_layers = len(hidden_states)
    fig, axes = plt.subplots(nrows=num_layers, ncols=1, figsize=(10, 3 * num_layers))
    
    for layer in range(num_layers):
        activation_values = hidden_states[layer][0, token_index, :].cpu().numpy()
        axes[layer].bar(range(len(activation_values)), activation_values)
        axes[layer].set_title(f'Layer: {layer} | Token: "{tokens[token_index]}"')
        axes[layer].set_xlabel('Neuron Index')
        axes[layer].set_ylabel('Activation Value')
    
    plt.tight_layout()
    plt.show()

    if extreme_nodes:
        print(f"\nExtreme nodes for token \"{tokens[token_index]}\" across all layers:")
        print(f"{'Layer':>6} {'Min Node':>10} {'Max Node':>10}")
        print("-" * 28)
        
        for layer in range(len(hidden_states)):
            min_node, max_node = find_extreme_nodes_for_specific_token(hidden_states, layer, token_index, Print=False)
            print(f"{layer:>6} {min_node:>10} {max_node:>10}")
        print()  # Add blank line after table
            


def find_extreme_nodes_all_tokens(hidden_states, layer):
    """
    Find extreme nodes for all tokens in a specific layer.
    
    Args:
        hidden_states (tuple): Hidden states from model output.
        layer (int): Layer index to analyze.
    
    Note:
        Requires 'tokens' variable to be defined in the calling scope.
    """
    print(f"\nExtreme nodes for all tokens in Layer {layer}:")
    print(f"{'Token':>15} {'Index':>6} {'Min Node':>10} {'Max Node':>10}")
    print("-" * 43)
    
    for i in range(len(tokens)):
        min_node, max_node = find_extreme_nodes_for_specific_token(hidden_states, layer, i, Print=False)
        token_display = tokens[i][:12] + "..." if len(tokens[i]) > 15 else tokens[i]
        print(f"{token_display:>15} {i:>6} {min_node:>10} {max_node:>10}")
    print()  # Add blank line after table


def steer_node_activation(layer_to_steer, node_to_steer, token_index, steering_coefficient):
    """
    Steer the activation of a specific node in a specific layer for a specific token.
    
    Args:
        layer_to_steer (int): Layer index to apply steering to.
        node_to_steer (int): Node index to steer.
        token_index (int): Token index to modify.
        steering_coefficient (float): Steering strength.
    
    Returns:
        tuple: (steered_embedding, steered_hidden_states).
    
    Note:
        Requires 'model' and 'encoded_input' variables to be defined in the calling scope.
    """

    def steering_hook(module, input, output):
        output[:, token_index, node_to_steer] += steering_coefficient
        return output

    hook_handle = model.encoder.layer[layer_to_steer].output.register_forward_hook(steering_hook)

    with torch.no_grad():
        steered_model_output = model(**encoded_input, output_hidden_states=True)

    hook_handle.remove()

    steered_hidden_states = steered_model_output.hidden_states
    steered_embedding = steered_hidden_states[-1][0, token_index, :]

    return steered_embedding, steered_hidden_states

'''
# Example usage token
# Import data from pkl file
tokenizer, hidden_states, all_texts_data = import_embedding_data_from_pkl("Test_export_embeddings.pkl", tokenizer=True, hidden_states=True, all_texts_data=True)  # Load the embeddings data
sample_text = all_texts_data['overview'][0]  # Use the first text from the loaded data
tokens = tokenizer.tokenize(sample_text)

# Plot activation values for a specific token
plot_activation_values_specific_token(hidden_states, 5, 3)  # Layer 5, Token index 3
plot_activation_values_all_layers(hidden_states, 3)  # Token index 3
find_extreme_nodes_all_tokens(hidden_states, 5)  # Layer 5

# Steer activation
steered_embedding, steered_hidden_states = steer_activation(layer_to_steer = 4, node_to_steer = 223, token_index = token_index, steering_coefficient = 100)

'''
###############################
# Sentences

def mean_pooling_from_hidden(hidden_state, attention_mask):
    """
    Perform mean pooling on the hidden state to get sentence embeddings.
    
    Args:
        hidden_state (torch.Tensor): Hidden state tensor of shape (batch_size, seq_len, hidden_size).
        attention_mask (torch.Tensor): Attention mask tensor.
    
    Returns:
        torch.Tensor: Mean-pooled sentence embeddings.
    """
    token_embeddings = hidden_state  # Shape: (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_text_activations_per_layer(hidden_states, attention_mask):
    """
    Compute the text activations for each layer by applying mean pooling.
    
    Args:
        hidden_states (tuple): Hidden states from all layers.
        attention_mask (torch.Tensor): Attention mask tensor.
    
    Returns:
        list: Text activations for each layer.
    """
    text_activations = []
    for layer in range(len(hidden_states)):
        layer_activations = mean_pooling_from_hidden(hidden_states[layer], attention_mask)
        text_activations.append(layer_activations)
    return text_activations

def plot_text_activations_per_layer(text_activations, all_texts_data, layers=None, text_index=0, extreme_nodes=True):
    """
    Plot the text activations for a specific text across specified layers.
    
    Args:
        text_activations (list): Text activations for each layer.
        all_texts_data (pd.DataFrame): Text metadata with title column.
        layers (None, int, tuple, or list, optional): Layers to plot. None for all layers. Defaults to None.
        text_index (int, optional): Index of text to analyze. Defaults to 0.
        extreme_nodes (bool, optional): Whether to find extreme nodes. Defaults to True.
    """
    selected_layers = []

    if layers is None:
        # Plot all layers
        selected_layers = list(range(len(text_activations)))
    elif isinstance(layers, int):
        # Single layer
        selected_layers = [layers]
    elif isinstance(layers, (tuple, list)):
        # Multiple specific layers
        selected_layers = list(layers)
    else:
        raise ValueError("layer must be None, int, tuple, or list")

    num_layers = len(selected_layers)

    # Handle single subplot case
    if num_layers == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))
        axes = [axes]  # Make it iterable
    else:
        fig, axes = plt.subplots(nrows=num_layers, ncols=1, figsize=(10, 3 * num_layers))
    
    title = all_texts_data['title'].iloc[text_index]
    title_display = title[:35] + "..." if len(title) > 38 else title
    text_label = f"({text_index}) {title_display}"

    for i, layer_idx in enumerate(selected_layers):
        activation_values = text_activations[layer_idx][text_index].cpu().numpy()
        axes[i].bar(range(len(activation_values)), activation_values)
        axes[i].set_title(f'Layer: {layer_idx} | Text: {text_label}')
        axes[i].set_xlabel('Neuron Index')
        axes[i].set_ylabel('Activation Value')
    
    plt.tight_layout()
    plt.show()

    if extreme_nodes:
        print(f"\nExtreme nodes for text {text_label} across selected layers:")
        print(f"{'Layer':>6} {'Min Node':>10} {'Max Node':>10}")
        print("-" * 28)

        for layer_idx in selected_layers:
            min_node, max_node = find_extreme_nodes_for_specific_text(text_activations, layer_idx, text_index, Print=False)
            print(f"{layer_idx:>6} {min_node:>10} {max_node:>10}")
        print()  # Add blank line after table
    


def compare_text_activations(text_activations, all_texts_data, text_indices=(0, 1), layers=None, extreme_nodes=True):
    """
    Compare the activations of two texts across specified layers.
    
    Args:
        text_activations (list): Text activations for each layer.
        all_texts_data (pd.DataFrame): Text metadata with title column.
        text_indices (tuple, optional): Indices of texts to compare. Defaults to (0, 1).
        layers (None, int, tuple, or list, optional): Layers to plot. None for all layers. Defaults to None.
        extreme_nodes (bool, optional): Whether to find extreme nodes. Defaults to True.
    """
    # Get titles for both texts
    title_1 = all_texts_data['title'].iloc[text_indices[0]]
    title_2 = all_texts_data['title'].iloc[text_indices[1]]
    title_1_display = title_1[:20] + "..." if len(title_1) > 23 else title_1
    title_2_display = title_2[:20] + "..." if len(title_2) > 23 else title_2
    
    labels = [f"({text_indices[0]}) {title_1_display}", f"({text_indices[1]}) {title_2_display}"]
    
    # Handle layer selection
    selected_layers = []
    if layers is None:
        # Plot all layers
        selected_layers = list(range(len(text_activations)))
    elif isinstance(layers, int):
        # Single layer
        selected_layers = [layers]
    elif isinstance(layers, (tuple, list)):
        # Multiple specific layers
        selected_layers = list(layers)
    else:
        raise ValueError("layers must be None, int, tuple, or list")

    num_layers = len(selected_layers)

    # Handle single subplot case
    if num_layers == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))
        axes = [axes]  # Make it iterable
    else:
        fig, axes = plt.subplots(nrows=num_layers, ncols=1, figsize=(10, 3 * num_layers))
    
    for i, layer_idx in enumerate(selected_layers):
        activation_values_1 = text_activations[layer_idx][text_indices[0]].cpu().numpy()
        activation_values_2 = text_activations[layer_idx][text_indices[1]].cpu().numpy()
        
        axes[i].bar(range(len(activation_values_1)), activation_values_1, alpha=0.5, label=labels[0], color='blue')
        axes[i].bar(range(len(activation_values_2)), activation_values_2, alpha=0.5, label=labels[1], color='red')
        
        axes[i].set_title(f'Layer: {layer_idx}')
        axes[i].set_xlabel('Neuron Index')
        axes[i].set_ylabel('Activation Value')
        axes[i].legend()
        
    plt.tight_layout()
    plt.show()

    if extreme_nodes:
        print(f"\nExtreme nodes comparison for texts ({text_indices[0]}) {title_1} and ({text_indices[1]}) {title_2}:")
        print(f"{'Layer':>6} {'T{0} Min':>8} {'T{0} Max':>8} {'T{1} Min':>8} {'T{1} Max':>8}".format(text_indices[0], text_indices[1]))
        print("-" * 40)

        for layer_idx in selected_layers:
            min_node_1, max_node_1 = find_extreme_nodes_for_specific_text(text_activations, layer_idx, text_indices[0], Print=False)
            min_node_2, max_node_2 = find_extreme_nodes_for_specific_text(text_activations, layer_idx, text_indices[1], Print=False)
            print(f"{layer_idx:>6} {min_node_1:>8} {max_node_1:>8} {min_node_2:>8} {max_node_2:>8}")
        print()  # Add blank line after table

def find_extreme_nodes_for_specific_text(text_activations, layer, text_index, Print=True):
    """
    Find the nodes with the least and most activation for a specific text in a specific layer.
    
    Args:
        text_activations (list): Text activations for each layer.
        layer (int): Layer index to analyze.
        text_index (int): Index of text to analyze.
        Print (bool, optional): Whether to print results. Defaults to True.
    
    Returns:
        tuple: (min_node, max_node) indices.
    """
    activation_values = text_activations[layer][text_index].cpu().numpy()
    min_node = activation_values.argmin()
    max_node = activation_values.argmax()
    min_value = activation_values[min_node]
    max_value = activation_values[max_node]

    if Print:
        title = all_texts_data['title'].iloc[text_index]
        title_display = title[:35] + "..." if len(title) > 38 else title
        text_label = f"({text_index}) {title_display}"
        print(f"\nLayer {layer} | Text {text_label}")
        print(f"{'Type':>10} {'Node':>8} {'Activation':>12}")
        print("-" * 32)
        print(f"{'Min':>10} {min_node:>8} {min_value:>12.4f}")
        print(f"{'Max':>10} {max_node:>8} {max_value:>12.4f}")

    return min_node, max_node


def find_extreme_nodes_for_all_texts(texts, text_activations, all_texts_data, layer):
    """
    Find extreme nodes for all texts in a specific layer.
    
    Args:
        texts (list): List of texts (not used in function but kept for compatibility).
        text_activations (list): Text activations for each layer.
        all_texts_data (pd.DataFrame): Text metadata with title column.
        layer (int): Layer index to analyze.
    
    Returns:
        tuple: (min_nodes, max_nodes) lists of indices.
    """
    min_nodes = [0]*len(texts)
    max_nodes = [0]*len(texts)

    print(f"\nExtreme nodes for all texts in Layer {layer}:")
    print(f"{'Text':>50} {'Min Node':>10} {'Max Node':>10}")
    print("-" * 72)
    
    for i in range(len(texts)):
        min_nodes[i], max_nodes[i] = find_extreme_nodes_for_specific_text(text_activations, layer, i, Print=False)
        title = all_texts_data['title'].iloc[i]
        title_display = title[:35] + "..." if len(title) > 38 else title
        text_label = f"({i}) {title_display}"
        print(f"{text_label:>50} {min_nodes[i]:>10} {max_nodes[i]:>10}")
    print()  # Add blank line after table
    return min_nodes, max_nodes

def plot_extreme_nodes_for_all_texts(min_nodes, max_nodes, layer):
    """
    Plot extreme nodes for all texts in a specific layer.
    
    Args:
        min_nodes (list): List of minimum node indices for each text.
        max_nodes (list): List of maximum node indices for each text.
        layer (int): Layer index being analyzed.
    """

    num_texts = np.arange(len(min_nodes))

    plt.plot(num_texts, min_nodes, "bo",  label = "Min_nodes")
    plt.plot(num_texts, max_nodes, "ro",  label = "Max_nodes")
    plt.xlabel("Text")
    plt.ylabel("Extreme Node")
    plt.title(f"Extreme Nodes per Text | Layer {layer}")
    plt.legend()
    plt.show()



# Example usage
if __name__ == "__main__":
    # Import data from pkl file
    data = import_embedding_data_from_pkl(
        "Test_export_embeddings.pkl",
        model=True, 
        tokenizer=True, 
        encoded_input=True, 
        hidden_states=True, 
        all_texts_data=True)
    
    model, tokenizer, encoded_input, hidden_states, all_texts_data = data  # Unpack in the order they were requested
    texts = all_texts_data['overview'].tolist()[:10]  # Use the overview column for text data

    feature = "War"
    layer_to_steer = 11
    steering_coefficient = 1
    steering_vector = import_steering_vector_from_pkl('steering_vector.pkl', feature, layer_to_steer=layer_to_steer)

    steered_model_output = create_steered_model_output(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector, verbose=False)
    steered_hidden_states = steered_model_output.hidden_states

    # Plots
    text_activations = get_text_activations_per_layer(hidden_states, encoded_input['attention_mask'])
    steered_text_activations = get_text_activations_per_layer(steered_hidden_states, encoded_input['attention_mask'])

    plot_text_activations_per_layer(text_activations, all_texts_data, layers=(0, 11), text_index=0)  # Plot for the first text
    plot_text_activations_per_layer(steered_text_activations, all_texts_data, layers=(0, 11), text_index=0)  # Plot for the first text after steering

    compare_text_activations(text_activations, all_texts_data, text_indices=(0, 1), layers=(0, 11))  # Compare activations for the first two texts
    compare_text_activations(steered_text_activations, all_texts_data, text_indices=(0, 1), layers=(0, 11))  # Compare activations for the first two texts after steering

    min_nodes, max_nodes = find_extreme_nodes_for_all_texts(texts, text_activations, all_texts_data, layer=11)  # Find extreme nodes for all texts in layer 11
    steered_min_nodes, steered_max_nodes = find_extreme_nodes_for_all_texts(texts, steered_text_activations, all_texts_data, layer=11)  # Find extreme nodes for all texts in steered layer 11

    plot_extreme_nodes_for_all_texts(min_nodes, max_nodes, layer=11)  # Plot extreme nodes for all texts in layer 11
    plot_extreme_nodes_for_all_texts(steered_min_nodes, steered_max_nodes, layer=11)  # Plot extreme nodes for all texts in steered layer 11   