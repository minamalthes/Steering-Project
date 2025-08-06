from Embeddings import import_embedding_data_from_pkl, mean_pooling
from Steering_vector import import_steering_vector_from_pkl
import torch
import torch.nn.functional as F

def create_steered_model_output(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector, verbose=True):
    """
    Steer the activation of a specific layer by adding a steering vector.
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
    Steers the activation of a specific node in a specific layer.
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
    Steers the activation of a specific neuron in a specific layer for a specific token.
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