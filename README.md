# project-4-ian-livingston
Alaska property NLP analysis (Metis project 4)

## Description
In this project, my third solo DS project and my first in the realm of unsuprevised NLP, I attempted to identify patterns by way of topic modeling in Tripadvisor reviews. I narrowed in on all properties in Alaska (~3,000), scraped reviews and review metadata using many iterations of a Selenium-powered function, pre-processed my corpus and set to pulling it apart. After a few unsuccessful attempts at topic modeling I pulled 11 reasonably evident topics (below) using CorEx.

## Topics identified by final CorEx model
- Arrival
- Room 
- Atmosphere 
- Surrounding area 
- Annoyances/quirks 
- Food
- Breakfast
- Check-in/check-out
- Touring Alaska
- Local scenery
- Fishing

## Data Used
I used data scraped and sourced from:

- Tripadvisor

## Tools Used
Additionally, I made use of the following Python modules in the course of this project:

- Scikit-learn
- NLTK
- SpaCy
- Numpy
- Pandas
- CorEx
- Vader
- BeautifulSoup
- Selenium
- Matplotlib
- Time
- Re
- Plotly
- Tableau
- Flask

## Possible impacts
The major takeaways were in the rhythms found in the unsupervised NLP space. Beyond that, the language of internet reviews proved to be difficult to parse and clean. I'm sure that tools exist to help with this effort, and I'm also sure that more work and more time in the space will naturally return more comfort navigating language and the intention behind it. Immediately I found it encouraging to see topics like the "Annoyances/quirks" trend linearly with latirude change, and the range of sentiment measured in reviews by Vader change (slightly) with the seasons in a seasonal destination.

There's a ton I would have liked to do with more time, and in NLP work in the future. That starts with text and corpus cleaning, which I feel I did not do sufficiently given the limited timeframe.

