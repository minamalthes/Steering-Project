import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import plotly.express as px
from sklearn.decomposition import PCA
import pickle as pickle
import plotly.graph_objects as go
import os


def set_model_and_tokenizer(model_name):
    """
    Set the model and tokenizer for text embeddings.
    
    Args:
        model_name (str): Name of the pre-trained model to load.
    
    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()  # Set the model to evaluation mode
    print(f"Model and tokenizer set for {model_name}")


    return model, tokenizer


def import_data(file_path, number_of_texts=1000):
    """
    Import data from a pkl file and preprocess column names.
    
    Args:
        file_path (str): Path to the pickle file containing the data.
        number_of_texts (int, optional): Maximum number of texts to process. Defaults to 1000.
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns ('title', 'genre', 'overview').
    
    Note:
        Expected input columns: 'Series_Title', 'Best_Fit_Genre', 'Overview', 'Genre'.
        Drops 'Genre' column and renames others to standard format.
    """
    # Open the pickle file 
    with open(file_path, 'rb') as f:
        all_texts_data = pickle.load(f)

    # Remove the "Genre" column and rename "Best fit genre" to "genre" and "Overview" to "overview"
    all_texts_data = all_texts_data.drop(columns=["Genre"])
    all_texts_data = all_texts_data.rename(columns={"Series_Title": "title", "Best_Fit_Genre": "genre", "Overview": "overview"})

    print("Data imported successfully.")
    print("Number of texts:", len(all_texts_data))
    print(all_texts_data.head())

    #texts = all_texts_data['overview'].tolist()[:number_of_texts]  # Use the overview column for text data

    return all_texts_data


def get_embeddings(model_name, texts, normalize=True):
    """
    Get embeddings for a list of texts using a pre-trained model.
    
    Args:
        model_name (str): Name of the pre-trained model to use.
        texts (list): List of text strings to embed.
        normalize (bool, optional): Whether to normalize embeddings. Defaults to True.
    
    Returns:
        tuple: (model, tokenizer, model_output, embeddings, encoded_input, hidden_states).
    """

    # Set model and tokenizer
    model, tokenizer = set_model_and_tokenizer(model_name)

    encoded_input = tokenizer(texts, padding = True, truncation = True, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True)

    # Calculate sentence embeddings
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize embeddings

    hidden_states = model_output.hidden_states  # Get hidden states from the model output
    
    print(f"Got embeddings for {len(texts)} texts using model: {model_name}")
    return model, tokenizer, model_output, embeddings, encoded_input, hidden_states


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on the model output to get sentence embeddings.
    
    Args:
        model_output: Model output containing token embeddings.
        attention_mask (torch.Tensor): Attention mask for the input tokens.
    
    Returns:
        torch.Tensor: Mean-pooled sentence embeddings.
    """
    token_embeddings = model_output[0]  # Shape: (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)


def show_hidden_states(hidden_states):
    """
    Display information about the hidden states from a transformer model.
    
    Args:
        hidden_states (tuple): Hidden states from all layers of the model.
    
    Note:
        Layer 0 contains embedding output, subsequent layers contain encoder outputs.
    """

    print(f"Number of transformer layers: {len(hidden_states) - 1}")
    print("Note: Layer 0 is the embedding output (before first encoder layer)\n")

    for i, layer in enumerate(hidden_states):
        if i == 0:
            print(f"Embedding output (hidden_states[0]) shape: {layer.shape}")
        else:
            print(f"Encoder Layer {i-1}  --> output in hidden_states[{i}], Shape: {layer.shape}")


def export_embeddings_to_pkl(model_name, model, tokenizer, model_output, embeddings, encoded_input, hidden_states, all_texts_data, file_path):
    """
    Export model components and embeddings to a pickle file.
    
    Args:
        model_name (str): Name of the model used.
        model: The transformer model object.
        tokenizer: The tokenizer object.
        model_output: Output from the model forward pass.
        embeddings (torch.Tensor): Computed sentence embeddings.
        encoded_input (dict): Tokenized input data.
        hidden_states (tuple): Hidden states from all layers.
        all_texts_data (pd.DataFrame): Original text metadata.
        file_path (str): Path where to save the pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump({
            'model_name': model_name,
            'model': model,  # Save the model itself
            'tokenizer': tokenizer,  # Save the tokenizer
            'model_output': model_output,
            'embeddings': embeddings,
            'encoded_input': encoded_input,
            'hidden_states': hidden_states,
            'all_texts_data': all_texts_data
        }, f)
    print(f"Embeddings exported to {file_path}")


def print_file_size(path):
    """
    Print the file size in GB for a given file path.
    
    Args:
        path (str): Path to the file to check.
    """
    size_bytes = os.path.getsize(path)
    size_gb = size_bytes / (1024 ** 3)
    print(f"Importing {size_gb:.2f} GB data from file {path}...")

def import_embedding_data_from_pkl(file_path, model_name=False, model=False, tokenizer=False, model_output=False, embeddings=False, 
                                   encoded_input=False, hidden_states=False, all_texts_data=False):
    """
    Import specific components from a pickle file containing embedding data.
    
    Args:
        file_path (str): Path to the pickle file to load.
        model_name (bool, optional): Whether to load model name. Defaults to False.
        model (bool, optional): Whether to load the model. Defaults to False.
        tokenizer (bool, optional): Whether to load the tokenizer. Defaults to False.
        model_output (bool, optional): Whether to load model output. Defaults to False.
        embeddings (bool, optional): Whether to load embeddings. Defaults to False.
        encoded_input (bool, optional): Whether to load encoded input. Defaults to False.
        hidden_states (bool, optional): Whether to load hidden states. Defaults to False.
        all_texts_data (bool, optional): Whether to load text metadata. Defaults to False.
    
    Returns:
        tuple or single item: Requested components in the order they were requested.
        Returns single item if only one requested, tuple if multiple, None if none.
    """

    print_file_size(file_path)

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data imported from {file_path}")
    results = []
    
    if model_name is True:
        model_name_data = data.get('model_name', None)
        print(f"Model name: {model_name_data}")
        results.append(model_name_data)
    if model is True:
        model_data = data.get('model', None)
        print("Model loaded successfully.")
        results.append(model_data)
    if tokenizer is True:
        tokenizer_data = data.get('tokenizer', None)
        print("Tokenizer loaded successfully.")
        results.append(tokenizer_data)
    if model_output is True:
        model_output_data = data.get('model_output', None)
        print("Model output loaded successfully.")
        results.append(model_output_data)
    if embeddings is True:
        embeddings_data = data.get('embeddings', None)
        print("Embeddings loaded successfully.")
        results.append(embeddings_data)
    if encoded_input is True:
        encoded_input_data = data.get('encoded_input', None)
        print("Encoded input loaded successfully.")
        results.append(encoded_input_data)
    if hidden_states is True:
        hidden_states_data = data.get('hidden_states', None)
        print("Hidden states loaded successfully.")
        results.append(hidden_states_data)
    if all_texts_data is True:
        all_texts_data_data = data.get('all_texts_data', None)
        print("All texts data loaded successfully.")
        results.append(all_texts_data_data)

    # Return single item if only one requested, tuple if multiple, or None if none requested
    if len(results) == 1:
        return results[0]
    elif len(results) > 1:
        return tuple(results)
    else:
        return None
    

# Example usage

if __name__ == "__main__":
    # Import the data
    #all_texts_data = import_data("path/to/your/data.pkl", number_of_texts=1000) #../imdb_top_1000_with_best_fit_genre.pkl
    all_texts_data = import_data("../imdb_top_1000_with_best_fit_genre.pkl", number_of_texts=1000) #../imdb_top_1000_with_best_fit_genre.pkl
    texts = all_texts_data['overview'].tolist()[:1000]  # Use the overview column for text data

    # Choose a model
    model_name = "sentence-transformers/all-MiniLM-L12-v2"

    # Get the embeddings
    model, tokenizer, model_output, embeddings, encoded_input, hidden_states = get_embeddings(model_name, texts)
    show_hidden_states(hidden_states)

    # Export the embeddings to a pkl file
    #export_embeddings_to_pkl(model_output, embeddings, encoded_input, all_texts_data, "path/to/your/embeddings.pkl") #Test_export_embeddings.pkl
    export_embeddings_to_pkl(model_name, model, tokenizer, model_output, embeddings, encoded_input, hidden_states, all_texts_data, "Test_export_embeddings.pkl") #Test_export_embeddings.pkl

    #import_embedding_data_from_pkl("Test_export_embeddings.pkl", model_name=True, model=True, tokenizer=True, model_output=True, 
    #                               embeddings=True, encoded_input=True, hidden_states=True, all_texts_data=True)

