from chat_downloader import ChatDownloader
from itertools import islice

class ChatCommentsDownloader:
    def __init__(self, video_url: str, max_comments: int = 50) -> None:
        self.url = video_url
        self.max_comments = max_comments

        self.downloader = ChatDownloader()

    def download_comments(self):

        chat = self.downloader.get_chat(self.url)

        all_comments = []
    
        for message in islice(chat, self.max_comments):
            comment = {}
            comment['text'] = message.get('message')
            comment['author'] = message.get('author').get('name')

            all_comments.append(comment)

        return all_comments