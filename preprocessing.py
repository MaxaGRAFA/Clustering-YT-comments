from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer

from itertools import islice

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.manifold import TSNE
import umap

import random

import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

class CommentsPreprocessor:
    def __init__(self, video_url: str) -> None:
        self.model = SentenceTransformer('D:/PythonFiles/CommentsFromTwitch/model/intfloat_multilingual-e5-base')
        self.video_url = video_url

    def start(self, comments: list) -> pd.DataFrame:
        df = self.preprocess_comments(comments)
        df = self.reduce_embeddings(df)
        df = self.paint_clusters(df)
        
        return df

    def preprocess_comments(self, comments: list) -> pd.DataFrame:
        max_threads = config['Max_Threads']

        data_list = []
        with ThreadPoolExecutor(max_threads) as executor:
            futures = {executor.submit(self.process_comment, comment): comment for comment in islice(comments, config['Max_Comments'])}
            for future in as_completed(futures):
                data_list.append(future.result())
                
                
        df = pd.DataFrame(data_list)
        df['Embeddings'] = np.array(df['Embeddings'])

        return df
        
    #encode comment into embeddings
    def process_comment(self, comment: dict) -> dict:
        input_texts = ['query: {}'.format(comment['text'])]
        embeddings = self.model.encode(input_texts, normalize_embeddings=True)

        return {'Embeddings': embeddings[0], 'Author': comment['author'], 'Text': comment['text']}

    #reduce so that you can display the data on the screen
    def reduce_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        if config['Visualization_Method'] =='UMAP': 
            algorithm = umap.UMAP(n_components=2)
        elif config['Visualization_Method'] =='TSNE':
            algorithm = TSNE(n_components=2)

        all_embeddings = np.array(df['Embeddings'].tolist())

        X_embedded = algorithm.fit_transform(all_embeddings)

        df['Reduced_Embeddings'] = list(X_embedded)

        return df
    
    #Give each cluster its own color
    def paint_clusters(self, df: pd.DataFrame) -> pd.DataFrame:

        labels = self.clustering(df)

        unique_values = list(set(labels))
    
        color_mapping = {}

        labels_df = pd.DataFrame({'labels': labels})

        for num in unique_values:
            color_mapping[num] = self.random_color()

        df['Color'] = labels_df['labels'].map(color_mapping)

        return df

    #Cluster all comments, here I use leiden-algorithm because it works best
    def clustering(self, df: pd.DataFrame) -> pd.Categorical:
        all_embeddings = np.array(df['Embeddings'].tolist())

        adata = sc.AnnData(all_embeddings)

        sc.pp.neighbors(adata, n_neighbors=config['n_neighbors'], use_rep='X')
        
        sc.tl.leiden(adata, resolution=config['Resolution'], flavor="leidenalg")

        return adata.obs['leiden'].values
            

    def random_color(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))