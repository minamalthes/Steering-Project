from PCA import do_kmeans, create_hover_data

import torch
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go

def plot_tsne_fixed_kmeans(original_embeddings, steered_embeddings, all_texts_data, info_string, text_range=None, n_clusters=5, 
                           lines=True, Write = False, Show = True, return_fig = False):
    """
    Plot t-SNE visualization of original vs steered embeddings with K-means clustering.
    """
    # Apply slicing if requested
    if text_range is not None:
        s = slice(*text_range)
        all_texts_data = all_texts_data.iloc[s]
        original_embeddings = original_embeddings[s]
        steered_embeddings = steered_embeddings[s]

    hover_data_original, hover_data_steered = create_hover_data(all_texts_data, len(original_embeddings))

    # Fit K-Means on original embeddings
    original_labels = do_kmeans(original_embeddings, n_clusters)
    combined_embeddings = torch.cat([original_embeddings, steered_embeddings], dim=0).cpu().numpy()

    # Perform t-sne
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate="auto", early_exaggeration=5.0, n_iter=5000, init="pca")
    X_tsne = tsne.fit_transform(combined_embeddings)

    # Split transformed points
    tsne_original = X_tsne[:len(original_embeddings)]
    tsne_steered = X_tsne[len(original_embeddings):]

    # Create DataFrame
    df_original = pd.DataFrame(tsne_original, columns=['t-SNE 1', 't-SNE 2'])
    df_original['Cluster'] = original_labels.astype(str)
    df_original['Title'] = hover_data_original
    df_original['Type'] = 'Original'

    df_steered = pd.DataFrame(tsne_steered, columns=['t-SNE 1', 't-SNE 2'])
    df_steered['Cluster'] = original_labels.astype(str) # Keep the same clusters
    df_steered['Title'] = hover_data_steered
    df_steered['Type'] = 'Steered'

    df = pd.concat([df_original, df_steered], ignore_index=True)

    fig = px.scatter(
        df,
        x='t-SNE 1',
        y='t-SNE 2',
        color='Cluster',
        symbol='Type',
        hover_name='Title',
        hover_data={'t-SNE 1': ':.3f', 't-SNE 2': ':.3f', 'Cluster': True, 'Type': True},
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    # Lower opacity for original points
    for trace in fig.data:
        if 'Original' in trace.name:
            trace.opacity = 0.3  # Transparent for original
        else:
            trace.opacity = 1.0  # Solid for steered

    if lines:
        # Add lines between original and steered points
        for i in range(len(df_original)):
            fig.add_trace(go.Scatter(
                x=[df_original.iloc[i]['t-SNE 1'], df_steered.iloc[i]['t-SNE 1']],
                y=[df_original.iloc[i]['t-SNE 2'], df_steered.iloc[i]['t-SNE 2']],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='none',
            ))

    fig.update_layout(title=f't-SNE Projection: Original vs. Steered Embeddings {info_string}'
                    , width=900, height=700)

    if Show:
        fig.show()
    if Write:
        fig.write_html(f"t-SNE Projection: {info_string}.html")

    if return_fig:
        return fig