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

<div style="display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0;">

**Python**  
![Python](https://www.python.org/static/community_logos/python-logo-master-v3-TM-flattened.png)

**Flask**  
![Flask](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Flask_logo.svg/500px-Flask_logo.svg.png)

**Pandas**  
![Pandas](https://pandas.pydata.org/static/img/pandas_mark.svg)

**scikit-learn**  
![scikit-learn](https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/320px-Scikit_learn_logo_small.svg.png)

**Redis**  
![Redis](https://redis.io/wp-content/uploads/2024/04/Redis_Logo.svg)

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
