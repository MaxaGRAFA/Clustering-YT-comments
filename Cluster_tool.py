from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer

from youtube_comment_downloader import * 
from pytube import YouTube

from itertools import islice

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.manifold import TSNE
import umap

import plotly.express as px

import random

import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

class Analytic_tool:
    def __init__(self, video_url: str) -> None:
        self.model = SentenceTransformer('intfloat/multilingual-e5-base')
        self.downloader = YoutubeCommentDownloader()

        self.video_url = video_url
    
    #Download comments and process them
    def download_comments(self) -> pd.DataFrame:
        comments = self.downloader.get_comments_from_url(self.video_url, sort_by=SORT_BY_POPULAR)

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
        embeddings = self.model.encode(input_texts, normalize_embeddings=True, show_progress_bar=True)

        return {'Embeddings': embeddings[0], 'Author': comment['author'], 'Text': comment['text']}

    #reduce so that you can display the data on the screen
    def Reduce_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if config['Visualization_Method'] =='UMAP': 
            algorithm = umap.UMAP(n_components=2)
        elif config['Visualization_Method'] =='TSNE':
            algorithm = TSNE(n_components=2)

        all_embeddings = np.array(df['Embeddings'].tolist())

        X_embedded = algorithm.fit_transform(all_embeddings)

        df['Reduced_Embeddings'] = list(X_embedded)

        return df
    
    
    #cluster all comments, here I use leiden-algorithm because it works best
    def clustering(self, df: pd.DataFrame) -> pd.Categorical:
        all_embeddings = np.array(df['Embeddings'].tolist())

        adata = sc.AnnData(all_embeddings)

        sc.pp.neighbors(adata, n_neighbors=config['n_neighbors'], use_rep='X')
        
        sc.tl.leiden(adata, resolution=config['Resolution'])

        return adata.obs['leiden'].values
    

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
    
    def random_color(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


    def visualize(self, df:pd.DataFrame) -> None:
        df['X'] = df['Reduced_Embeddings'].apply(lambda x: x[0])
        df['Y'] = df['Reduced_Embeddings'].apply(lambda x: x[1])

        video_title = self.get_title()

        fig = px.scatter(data_frame=df,
                        x='X',
                        y='Y',
                        color='Color', 
                        title=f'{video_title}', hover_data={'X': True, 'Y': True, 'Color': False, 'Text': True, 'Author': True})
        
        if config['Showlegend'] == False:
            fig.update_layout(showlegend=False)

        fig.show()
    
    def get_title(self) -> str:
        yt = YouTube(self.video_url)
        video_title = yt.title

        return video_title
    
    #add indents to the text to make it easier to read
    def text_transformation(self, df: pd.DataFrame) -> None:
        df['Text'] = df['Text'].apply(lambda x: "<br>".join(self.split_comment(x)))
        return df

    # dividing comments into small parts to make them readable
    def split_comment(self, comment):
        MAX_LINE_LENGTH = 120

        current_chunk = ""
        words = comment.split()
        transformed_lines = []

        for word in words:
            if len(current_chunk) + len(word)  <= MAX_LINE_LENGTH:
                current_chunk += word + " "
            else:
                transformed_lines.append(current_chunk.strip())
                current_chunk = word + " "

        transformed_lines.append(current_chunk.strip())
        return transformed_lines

    #function to start the entire project
    def start(self) -> None:
        df = self.download_comments()
        df = self.Reduce_embeddings(df)
        df = self.paint_clusters(df)
        df = self.text_transformation(df)
        self.visualize(df)
        
if __name__ == "__main__":

    your_url = input("Write your video url here: ")

    tool = Analytic_tool(video_url=your_url)
    tool.start()
