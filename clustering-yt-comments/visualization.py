
import plotly.express as px

from pytube import YouTube

import pandas as pd

import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

class VisualizationTool:
    def __init__(self, video_url: str) -> None:        
        self.video_url = video_url

    def visualize(self, df: pd.DataFrame) -> None:
        df = self.text_transformation(df)

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

    #splitting the comment text to the lists
    def split_comment(self, comment: str) -> list:
        max_line_length = 120

        current_chunk = ""
        words = comment.split()
        transformed_lines = []

        for word in words:
            if len(current_chunk) + len(word)  <= max_line_length:
                current_chunk += word + " "
            else:
                transformed_lines.append(current_chunk.strip())
                current_chunk = word + " "

        transformed_lines.append(current_chunk.strip())

        return transformed_lines

    #add indents to the text to make it easier to read
    def text_transformation(self, df: pd.DataFrame) -> None:
        df['Text'] = df['Text'].apply(lambda x: "<br>".join(self.split_comment(x)))
        return df
    
    def get_title(self) -> str:
        yt = YouTube(self.video_url)
        video_title = yt.title

        return video_title