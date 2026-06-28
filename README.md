# Anime Content-Based Recommendation System

This project builds a content-based anime recommendation system using anime metadata such as genre, anime type, and episode category.

The recommender suggests anime similar to a selected title by creating metadata-based content profiles, converting them into TF-IDF vectors, and ranking similar anime using cosine similarity.

Ratings, member count, and user rating statistics are shown for interpretation and decision support only. They are not used as the core similarity engine.

---

## Project objective

The objective is to build a content-based recommender system that answers:

> Which anime titles are most similar to the anime selected by the user based on content metadata?

The project also includes a Streamlit app where users can select an anime, filter recommendations, view similar titles, and explore dataset insights through a dashboard.

---

## Dataset

The dataset can be downloaded from this link: [Click here](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/data) 

Dataset files used:

```text
anime.csv
rating.csv
```

### anime.csv

The anime dataset contains:

* 12,294 anime records
* 7 columns
* Anime metadata such as title, genre, type, episodes, rating, and member count

Main columns:

```text
anime_id, name, genre, type, episodes, rating, members
```

### rating.csv

The rating dataset contains:

* 7,813,737 rating records
* 3 columns
* User-anime rating interactions

Main columns:

```text
user_id, anime_id, rating
```

In `rating.csv`, a rating value of `-1` means the user watched or interacted with the anime but did not provide an actual rating.

---

## Problem type

This is a recommendation system project.

The final recommender is content-based, not collaborative filtering.

The anime dataset is used as the main source for recommendation features. The rating dataset is used only for supporting EDA and anime-level aggregated statistics such as average user rating, valid rating count, total interactions, and unrated count.

---

## Assignment coverage

| Requirement                     | Covered in project                                                                                        |
| ------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Load dataset                    | Loaded `anime.csv` and `rating.csv` using Pandas                                                          |
| Initial inspection              | Checked shape, columns, data types, missing values, duplicates, and summary statistics                    |
| EDA and visualizations          | Performed analysis on anime type, genres, ratings, members, episodes, and user rating behavior            |
| Distribution and skewness check | Checked skewness for numerical columns such as episodes, rating, members, interactions, and rating counts |
| Transformations if required     | Converted episode count into an interpretable episode category for recommendation logic                   |
| Rating data cleaning            | Removed duplicate rating row and separated valid ratings from unrated interactions                        |
| Rating data aggregation         | Aggregated user ratings at anime level before merging                                                     |
| Safe merge                      | Merged aggregated rating statistics with anime data without increasing row count                          |
| Feature engineering             | Created content profiles using genre, anime type, and episode category                                    |
| Vectorization                   | Used TF-IDF vectorization on content profiles                                                             |
| Similarity calculation          | Used cosine similarity to identify similar anime                                                          |
| Final recommender model         | Selected TF-IDF Vectorization + Cosine Similarity as the final content-based recommendation approach      |
| Approach comparison             | Compared content-based recommendation against popularity-based, rating-based, and collaborative filtering approaches conceptually; selected content-based TF-IDF + cosine similarity because it matches the assignment scope                                                     |
| Recommendation testing          | Tested recommendations using multiple anime titles                                                        |
| Streamlit app                   | Built an interactive app with recommendation and dataset dashboard tabs                                   |
| Streamlit app readiness         | Saved required artifacts for Streamlit app usage                                                          |

---

## Data cleaning and preprocessing

The following preprocessing steps were performed:

* Checked missing values in both datasets
* Confirmed that `anime.csv` had missing values in `genre`, `type`, and `rating`
* Confirmed that `rating.csv` had no missing values
* Removed 1 duplicate row from `rating.csv`
* Treated `rating = -1` as unrated interaction
* Created a valid rating dataset by excluding `rating = -1`
* Aggregated rating data by `anime_id`
* Merged aggregated rating statistics with anime metadata using a safe left merge
* Renamed columns for clarity:

  * `name` → `anime_name`
  * `type` → `anime_type`
  * `rating` → `anime_rating`
  * `members` → `member_count`
* Cleaned HTML entities in anime names
* Converted episode count into numerical form
* Created episode category for content-based similarity

---

## Feature engineering

The recommender uses the following content features:

```text
genre
anime_type
episode_category
```

These features are combined into a single content profile for each anime.

Example content profile:

```text
drama romance school supernatural movie single episode
```

The content profile is then transformed using TF-IDF vectorization.

---

## Recommendation approach

The app recommends anime by comparing metadata-based content profiles created from genre, anime type, and episode category.
These profiles are converted into TF-IDF vectors, and cosine similarity is used to identify anime with the closest content pattern.

Ratings and popularity are shown only to help interpret the results; they are not used as the core similarity engine.

---

## Selected recommender model

The final selected recommendation approach is:

```text
TF-IDF Vectorization + Cosine Similarity
```

The recommender creates a metadata-based content profile for each anime using:

```text
genre + anime_type + episode_category
```

These content profiles are converted into TF-IDF vectors. Cosine similarity is then used to compare the selected anime against other anime and return the most similar titles.

### Final recommender configuration

| Component                 | Selected approach                                                                |
| ------------------------- | -------------------------------------------------------------------------------- |
| Recommendation type       | Content-based recommendation                                                     |
| Main input features       | Genre, anime type, episode category                                              |
| Text/vector method        | TF-IDF Vectorizer                                                                |
| Similarity method         | Cosine similarity                                                                |
| Ranking logic             | Highest content similarity score                                                 |
| Supporting display fields | Anime rating, average user rating, member count, valid rating count              |
| App optimization          | Similarity calculated on demand instead of saving a full dense similarity matrix |

This approach was selected because it directly matches the assignment requirement to create a content-based recommender system and provides explainable recommendations based on anime metadata.

---

## Approach comparison and selection rationale

Different recommendation approaches were considered, but the final project intentionally uses a content-based method.

| Approach                            | Used in final project? | Reason                                                                                                                |
| ----------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Popularity-based recommendation     | No                     | Would recommend generally popular anime, but not necessarily anime similar to the selected title                      |
| Rating-based recommendation         | No                     | Ratings help interpret quality but do not explain content similarity                                                  |
| Collaborative filtering             | No                     | Not selected because the assignment specifically asks for a content-based recommender system                          |
| User-user recommendation            | No                     | Requires user behavior similarity and would shift the project away from content-based recommendation                  |
| Item-item collaborative filtering   | No                     | Would use user rating patterns instead of anime metadata                                                              |
| CountVectorizer + cosine similarity | Possible alternative   | Can work, but TF-IDF gives better weighting by reducing the dominance of very common genre terms                      |
| TF-IDF + cosine similarity          | Yes                    | Selected because it creates explainable similarity scores from anime metadata and aligns directly with the assignment |

The selected model is therefore not a supervised machine learning model with accuracy, precision, recall, or F1-score. It is a recommendation model evaluated through content relevance, shared metadata, similarity scores, and sanity checks on sample anime recommendations.

---

## Final model output example

For the selected anime:

```text
Kimi no Na wa.
Genre: Drama, Romance, School, Supernatural
Type: Movie
Episode category: Single episode
```

The recommender returns similar anime such as:

```text
Aura: Maryuuin Kouga Saigo no Tatakai
Harmonie
Wind: A Breath of Heart OVA
Wind: A Breath of Heart (TV)
Air Movie
```

These recommendations are relevant because they share similar content patterns such as drama, romance, school, supernatural themes, and short or single-episode formats.


---

## Why this is content-based recommendation

This project does not use:

* User-user collaborative filtering
* Item-item collaborative filtering based on user ratings
* Matrix factorization
* User-specific recommendation history

Instead, recommendations are generated using anime metadata. This keeps the solution aligned with the assignment requirement to create a content-based recommender system.

---

## Streamlit app features

The Streamlit app includes two main tabs.

### 1. Recommend Anime

Users can:

* Search and select an anime
* Choose the number of recommendations
* Filter recommendations by anime type
* Filter recommendations by minimum anime rating
* View the selected anime profile
* Generate similar anime recommendations
* View similarity scores and supporting details
* Expand the recommendation output as a table

The recommendation output includes:

* Anime name
* Similarity score
* Genre
* Anime type
* Episode category
* Anime rating
* Average user rating
* Member count
* Valid rating count

### 2. Dataset Dashboard

The dashboard includes:

* Total anime titles
* Unique genre tags
* Average anime rating
* Total member count
* Anime type distribution
* Episode category distribution
* Top 10 genre tags
* Anime rating distribution
* Top 10 anime by member count
* Top 10 anime by valid rating count
* Data insights for interpretation

* View the Streamlit App here: [Click here](https://anime-content-recommendation-system.streamlit.app/) 

---

## Key insights

* TV titles dominate the anime type distribution, while single-episode and short-series titles are major episode-profile groups.
* Genre is the strongest content signal for recommendation.
* Episode category helps avoid comparing single-episode movies too aggressively with long-running series.
* Member count shows popularity but is not the same as content similarity.
* Ratings are useful for interpretation, but they are not used as the core similarity signal.
* Anime with many valid user ratings have more reliable audience feedback than titles with very few ratings.

---

## Saved artifacts

The notebook saves the following artifacts for Streamlit app usage:

```text
artifacts/
├── anime_app_data.pkl
├── anime_recommender_pipeline.pkl
├── anime_title_lookup.pkl
├── artifact_metadata.json
├── tfidf_matrix.pkl
└── tfidf_vectorizer.pkl
```

The app uses saved artifacts to ensure the Streamlit recommendation logic remains consistent with the notebook.

---

## Project structure

```text
Anime-content-recommendation-system/
│
├── app.py
├── code.ipynb
├── anime.csv (not uploaded to GitHub as it can be downloaded from the provided Kaggle link)
├── rating.csv (not uploaded to GitHub as it can be downloaded from the provided Kaggle link)
├── requirements.txt
│
├── assets/
│   └── Anime_banner.png
│
└── artifacts/
    ├── anime_app_data.pkl
    ├── anime_recommender_pipeline.pkl
    ├── anime_title_lookup.pkl
    ├── artifact_metadata.json
    ├── tfidf_matrix.pkl
    └── tfidf_vectorizer.pkl
```

---

## Limitations

This project recommends anime based only on available metadata such as genre, anime type, and episode category.
It does not personalize recommendations based on individual user history.

Anime with rare genres, missing metadata, or limited content information may receive fewer strong recommendation matches.
Ratings and popularity are included for interpretation but are not used as the main similarity signal.

---

## Future improvements

Possible future improvements include:

* Add anime poster images using a reliable anime metadata API
* Add more metadata, such as synopsis, studio, source, or release year, if available
* Add user-personalized recommendations as a separate collaborative filtering extension
* Compare content-based recommendations with collaborative filtering in a future advanced version
