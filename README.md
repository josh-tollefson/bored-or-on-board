# Bored, or on Board?
A project build to deliver focused feedback for board game developers

## Requirements
This program works on Anaconda Python v.3.7.3
> conda install --yes python==3.7.3

The following packages are also required

> pip install -r requirements.txt

## Data Scraping

Data scraping first needs to be done following the Board Game Scraper at: https://gitlab.com/recommend.games/board-game-scraper
by Markus Shepherd

> pip install board-game-scraper

This will return a JSON file (e.g., example.jl) containing user comments and board games. You need about 20000 comments to run some of the models from scratch

## Files 

chunk_files.sh: Break up example.jl into smaller JSON files 

pseudolabel.py: Perform pseudolabeling of comments using keywords related to board game related categories

run_categorezation.py: Run binary classification of comment categorization using example.jl (or whatever JSON file you scraped)

run_sentiment.py: Run binary classification of comment sentiment using example.jl (or whatever JSON file you scraped)

run_streamlit.py: Run streamlit web app

utils.py: Various utiliy functions (saving & loading files)

wordprocess.py: Various NLP functions

