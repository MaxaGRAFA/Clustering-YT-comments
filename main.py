from preprocessing import CommentsPreprocessor
from visualization import VisualizationTool

from download_chat import ChatCommentsDownloader
from youtube_comment_downloader import * 

from pytube import YouTube

import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

class AnalyticTool:
    def __init__(self, video_url: str) -> None:
        self.video_url = video_url

    def download_comments(self) -> list:
        if self.isVideoLiveStream():
            comments = ChatCommentsDownloader(self.video_url,  config['Max_Comments']).download_comments()
        else:
            comments = list(YoutubeCommentDownloader().get_comments_from_url(self.video_url, sort_by=SORT_BY_POPULAR))
            
        return comments

    def isVideoLiveStream(self) -> bool:
        yt =  YouTube(self.video_url)
        return yt.vid_info.get('videoDetails').get('isLiveContent')

    def start(self) -> None:
        preprocessor = CommentsPreprocessor(self.video_url)
        visualization = VisualizationTool(self.video_url)

        df = preprocessor.start(self.download_comments())
        visualization.visualize(df)
        

if __name__ == "__main__":
    video_url = input('Enter video url: ')
    analytic_tool = AnalyticTool(video_url)
    analytic_tool.start()




    
    