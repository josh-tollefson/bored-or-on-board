# Bored, or on Board?
A project built to deliver focused feedback for board game developers.

Board games are a multi-billion dollar industry nowadays and the number of board games published each year has grown tremendously since the early 2000s

![boarg-game-numbers](./bg-per-year.png)

## Requirements
This program works on Anaconda Python v.3.7.3
> conda install --yes python==3.7.3

The following packages are also required

> pip install -r requirements.txt

## Data Scraping

Data scraping first needs to be done following the Board Game Scraper at: https://gitlab.com/recommend.games/board-game-scraper
by Markus Shepherd

> pip install board-game-scraper

This will return a JSON file (e.g., example.jl) containing user comments and board games. 

## Files 

chunk_files.sh: Break up example.jl into smaller JSON files 

plotting.py: Useful visuzalizations (comparing model performance, word feature importance, etc)

pseudolabel.py: Perform pseudolabeling of comments using keywords related to board game related categories

run_categorezation.py: Run binary classification of comment categorization using example.jl (or whatever JSON file you scraped)

run_sentiment.py: Run binary classification of comment sentiment using example.jl (or whatever JSON file you scraped)

run_streamlit.py: Run streamlit web app

utils.py: Various utiliy functions (saving & loading files)

wordprocess.py: Various NLP functions

