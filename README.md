# Content Engine: Intelligent Recommendation System

**A lightweight, efficient content recommendation engine built with Python, leveraging content-based filtering for personalized suggestions.**

![Content Engine Architecture](https://picsum.photos/id/1015/800/400)

## Overview

This content engine is production ready flask-based REST webservice powers personalized recommendations by analyzing textual content using **TF-IDF vectorization** and **cosine similarity**. It is designed for scenarios like article recommenders, product suggestion engines, news feeds, or any text-heavy content platform.

The system processes items (e.g., articles, products) into vector representations and finds the most similar ones in real-time, with caching support for high performance.


## Key Features

- **Content-Based Filtering**: Recommends items similar to user-interacted content using TF-IDF and cosine similarity.
- **Fast Similarity Search**: Powered by scikit-learn's optimized linear kernel.
- **Caching Layer**: Redis integration for storing precomputed vectors and frequent queries.
- **Data Handling**: Pandas for efficient loading, cleaning, and manipulation of content datasets.
- **Web Integration**: Flask-ready for easy API exposure (e.g., `/predict` endpoints).
- **Real-time & Batch Processing**: Suitable for both on-the-fly recommendations and offline model updates.

## Tech Stack

<div style="display: flex; flex-wrap: wrap; gap: 30px; margin: 30px 0; align-items: center;">

**Python**  
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge)

**Flask**  
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white&style=for-the-badge)

**Pandas**  
![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white&style=for-the-badge)

**scikit-learn**  
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white&style=for-the-badge)

**Redis**  
![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white&style=for-the-badge)

</div>

### Core Libraries in Action

```python
import pandas as pd
import time
import redis
from flask import current_app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
```
app.py contains the two endpoints:

/train -- calls engine.train() which precomputes item similarities based on their descriptions in a csv file using TF-IDF and cosine similarity.

/predict -- given an item_id, returns the precomputed 'most similar' items according to the cosine similarity.
