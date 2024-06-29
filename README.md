# Clustering YouTube comments

You can take and analyze or read the comments of some video collected in a category. Like this:

![68d57d2f-3910-4806-8485-8b1b2bcf7062-ezgif com-video-to-gif-converter](https://github.com/MaxaGRAFA/Clustering-YT-comments/assets/89744777/6936c311-9989-49c8-a841-c6eae13f297b)

You can also analyze comments from streams on YouTube (The broadcast must be finished)

# Setup

You must download the requirements:
```bash
pip install -r requirements.txt
```

Also, when you first start it, the model (multilingual-e5-base) for embeddings will be automatically downloaded. It weighs ~3 gb.

# How to use

You should run python file main.py and enter your video link in the console

Processing ~1000 comments can take about 2 minutes. Depends on your hardware.

Don't forget to change the number of comments in config.yml. Default is 150

You will also need to set max_threads to a suitable value for your system for fast execution

And you can also try changing Visualization_Method to "TSNE"

