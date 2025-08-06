from Embeddings import import_embedding_data_from_pkl, mean_pooling
from Steering_vector import import_steering_vector_from_pkl
import torch
import torch.nn.functional as F

def create_steered_model_output(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector, verbose=True):
    """
    Steer the activation of a specific layer by adding a steering vector.
    
    Args:
        model: The transformer model.
        encoded_input (dict): Tokenized input data.
        layer_to_steer (int): Layer index to apply steering to.
        steering_coefficient (float): Strength of the steering intervention.
        steering_vector (torch.Tensor): Steering vector to add to the layer output.
        verbose (bool, optional): Whether to print shape information. Defaults to True.
    
    Returns:
        model_output: Full model output with steered activations and hidden states.
    
    Note:
        Uses forward hooks to modify layer outputs during the forward pass.
    """
    def steering_hook(module, input, output):
        return output + steering_vector*steering_coefficient  # Add the steering vector

    hook_handle = model.encoder.layer[layer_to_steer].output.register_forward_hook(steering_hook)

    with torch.no_grad():
        steered_model_output = model(**encoded_input, output_hidden_states=True)

    hook_handle.remove()
    
    if verbose:
        print("Created steered model output with shape:", steered_model_output.hidden_states[layer_to_steer].shape)

    return steered_model_output  # Return the full model output for pooling

def create_steered_model_output_neuron(model, encoded_input, layer_to_steer, node_to_steer, steering_coefficient, verbose=True):
    """
    Steer the activation of specific node(s) in a specific layer.
    
    Args:
        model: The transformer model.
        encoded_input (dict): Tokenized input data.
        layer_to_steer (int): Layer index to apply steering to.
        node_to_steer (int or tuple): Single node index or tuple of node indices to steer.
        steering_coefficient (float or tuple): Single coefficient or tuple of coefficients for each node.
        verbose (bool, optional): Whether to print shape information. Defaults to True.
    
    Returns:
        model_output: Full model output with steered neuron activations and hidden states.
    
    Note:
        Supports steering multiple neurons simultaneously if tuples are provided for both parameters.
    """

    def steering_hook(module, input, output):
        if isinstance(node_to_steer, tuple) and isinstance(steering_coefficient, tuple):
            # If both are tuples, apply steering to multiple nodes
            for node, coeff in zip(node_to_steer, steering_coefficient):
                output[:, :, node] += coeff
        else: 
            output[:, :, node_to_steer] += steering_coefficient
        return output

    hook_handle = model.encoder.layer[layer_to_steer].output.register_forward_hook(steering_hook)

    with torch.no_grad():
        steered_model_output = model(**encoded_input, output_hidden_states=True)

    hook_handle.remove()

    if verbose:
        print("Created steered model output with shape:", steered_model_output.hidden_states[layer_to_steer].shape)
        
    return steered_model_output


def get_steered_embeddings_vector(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector, normalize=True, verbose=True):
    """
    Generate embeddings from model output with vector-based steering applied.
    
    Args:
        model: The transformer model.
        encoded_input (dict): Tokenized input data with attention mask.
        layer_to_steer (int): Layer index where steering is applied.
        steering_coefficient (float): Strength multiplier for the steering vector.
        steering_vector (torch.Tensor or dict): Vector to add to activations, or dict mapping layers to vectors.
        normalize (bool, optional): Whether to L2-normalize the embeddings. Defaults to True.
        verbose (bool, optional): Whether to print shape information. Defaults to True.
    
    Returns:
        torch.Tensor: Mean-pooled embeddings from steered model output.
    
    Note:
        Automatically extracts the correct steering vector if a dictionary is provided.
    """

    if isinstance(steering_vector, dict):
        steering_vector = steering_vector[layer_to_steer]

    steered_output = create_steered_model_output(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector, verbose=verbose)
    steered_embeddings = mean_pooling(steered_output, encoded_input['attention_mask'])
    if normalize:
        steered_embeddings = F.normalize(steered_embeddings, p=2, dim=1)

    if verbose:
        print(f"Created steered embeddings with shape: {steered_embeddings.shape}")
    return steered_embeddings

def get_steered_embeddings_neuron(model, encoded_input, layer_to_steer, node_to_steer, steering_coefficient, normalize=True, verbose=True):
    """
    Generate embeddings from model output with neuron-specific steering applied.
    
    Args:
        model: The transformer model.
        encoded_input (dict): Tokenized input data with attention mask.
        layer_to_steer (int): Layer index where steering is applied.
        node_to_steer (int or tuple): Single node index or tuple of node indices to steer.
        steering_coefficient (float or tuple): Single coefficient or tuple of coefficients for each node.
        normalize (bool, optional): Whether to L2-normalize the embeddings. Defaults to True.
        verbose (bool, optional): Whether to print shape information. Defaults to True.
    
    Returns:
        torch.Tensor: Mean-pooled embeddings from steered model output.
    
    Note:
        Supports steering multiple neurons simultaneously if tuples are provided.
    """
    steered_output = create_steered_model_output_neuron(model, encoded_input, layer_to_steer, node_to_steer, steering_coefficient, verbose=verbose)
    steered_embeddings = mean_pooling(steered_output, encoded_input['attention_mask'])
    if normalize:
        steered_embeddings = F.normalize(steered_embeddings, p=2, dim=1)

    if verbose:
        print(f"Created steered embeddings with shape: {steered_embeddings.shape}")
    return steered_embeddings

# Example usage
if __name__ == "__main__":
    # Import data
    data = import_embedding_data_from_pkl('Test_export_embeddings.pkl', model=True, encoded_input=True)
    model, encoded_input = data
    steering_vector = import_steering_vector_from_pkl('steering_vector.pkl', 'Love', layer_to_steer=11)
    steered_embeddings = get_steered_embeddings_vector(model, encoded_input, layer_to_steer=11, steering_coefficient=0.5, steering_vector=steering_vector, normalize=True)
    steered_embeddings_neuron = get_steered_embeddings_neuron(model, encoded_input, layer_to_steer=11, node_to_steer=(0, 42, 100), steering_coefficient=(0.5, 0.5, 0.5), normalize=True)