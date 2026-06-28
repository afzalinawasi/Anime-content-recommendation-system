from pathlib import Path
import json
import joblib
import re
import html as html_lib

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Anime Content-Based Recommendation System",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")
ASSETS_DIR = Path("assets")
BANNER_PATH = ASSETS_DIR / "Anime_banner.png"

PIPELINE_PATH = ARTIFACTS_DIR / "anime_recommender_pipeline.pkl"
ANIME_APP_DATA_PATH = ARTIFACTS_DIR / "anime_app_data.pkl"
TFIDF_VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = ARTIFACTS_DIR / "tfidf_matrix.pkl"
TITLE_LOOKUP_PATH = ARTIFACTS_DIR / "anime_title_lookup.pkl"
METADATA_PATH = ARTIFACTS_DIR / "artifact_metadata.json"


# ------------------------------------------------------------
# Colour palette
# ------------------------------------------------------------
BLUE = "#0B3D91"
NAVY = "#0B1F3A"
RED = "#FF4B5C"
YELLOW = "#FFD60A"
GREY_BG = "#F4F7FB"
GREY_CARD = "#FFFFFF"
GREY_LINE = "#D7DEE9"
TEXT = "#111827"
TEXT_SOFT = "#4B5563"


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: #F4F7FB;
            color: #111827;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }

        h1, h2, h3 {
            color: #111827;
        }

        .project-intro {
            background: linear-gradient(135deg, #0B1F3A 0%, #0B3D91 70%);
            color: white;
            padding: 1.15rem 1.35rem;
            border-radius: 18px;
            margin-top: 1rem;
            margin-bottom: 1.1rem;
            border: 1px solid rgba(255,255,255,0.15);
        }

        .project-intro h1 {
            color: white;
            margin-bottom: 0.2rem;
            font-size: 2rem;
        }

        .project-intro p {
            color: #E5E7EB;
            margin-bottom: 0;
            font-size: 1rem;
        }

        .metric-card {
            background: #FFFFFF;
            padding: 1.05rem 1.1rem;
            border-radius: 16px;
            border: 1px solid #D7DEE9;
            box-shadow: 0 8px 22px rgba(11, 31, 58, 0.06);
            min-height: 116px;
        }

        .metric-label {
            color: #4B5563;
            font-size: 0.83rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-value {
            color: #0B1F3A;
            font-size: 1.75rem;
            font-weight: 900;
            margin-top: 0.25rem;
        }

        .metric-help {
            color: #4B5563;
            font-size: 0.84rem;
            margin-top: 0.3rem;
        }

        .info-card {
            background: #FFFFFF;
            border: 1px solid #D7DEE9;
            border-radius: 16px;
            padding: 1.05rem 1.15rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 22px rgba(11, 31, 58, 0.05);
        }

        .recommendation-card {
            background: #FFFFFF;
            border: 1px solid #D7DEE9;
            border-left: 6px solid #FF4B5C;
            border-radius: 16px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 8px 22px rgba(11, 31, 58, 0.06);
        }

        .anime-title {
            color: #0B1F3A;
            font-size: 1.08rem;
            font-weight: 900;
            margin-bottom: 0.35rem;
        }

        .soft-text {
            color: #4B5563;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .tag {
            display: inline-block;
            background: #EEF4FF;
            color: #0B3D91;
            padding: 0.22rem 0.5rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 0.25rem;
            margin-bottom: 0.25rem;
        }

        .score-tag {
            display: inline-block;
            background: #FFF7CC;
            color: #7A5900;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            margin-right: 0.25rem;
            margin-bottom: 0.25rem;
        }

        .insight-box {
            background: #FFFFFF;
            border-left: 5px solid #FFD60A;
            border-top: 1px solid #D7DEE9;
            border-right: 1px solid #D7DEE9;
            border-bottom: 1px solid #D7DEE9;
            padding: 0.85rem 1rem;
            border-radius: 12px;
            margin-top: 0.7rem;
            margin-bottom: 1.3rem;
            color: #111827;
        }

        .model-note {
            background: #FFFFFF;
            border: 1px solid #D7DEE9;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            color: #4B5563;
            margin-top: 1rem;
        }

        div.stButton > button:first-child {
            background: #FF4B5C;
            color: white;
            border: 0;
            border-radius: 12px;
            padding: 0.65rem 1rem;
            font-weight: 800;
            width: 100%;
        }

        div.stButton > button:first-child:hover {
            background: #E23549;
            color: white;
            border: 0;
        }

        [data-testid="stMetricValue"] {
            color: #0B1F3A;
        }

        [data-testid="stTabs"] button {
            font-weight: 800;
        }

        hr {
            border: none;
            border-top: 1px solid #D7DEE9;
            margin: 1.25rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_artifact_file(path: Path):
    """
    Load artifacts saved from the notebook using joblib only.

    The notebook artifact cells save .pkl files using joblib.dump(),
    so the Streamlit app must load them with joblib.load().
    """
    try:
        return joblib.load(path)
    except Exception as error:
        raise RuntimeError(
            f"Could not load artifact '{path}' using joblib.load(). "
            f"Please regenerate the artifacts from code.ipynb using joblib.dump(). "
            f"Original error: {error}"
        ) from error


def load_json_file(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_title(title: str) -> str:
    title = html_lib.unescape(str(title))
    title = title.lower().strip()
    title = re.sub(r"\s+", " ", title)
    return title


def safe_text(value, default: str = "Not available") -> str:
    if pd.isna(value):
        return default
    value = html_lib.unescape(str(value))
    return html_lib.escape(value)


def format_number(value) -> str:
    if pd.isna(value):
        return "Not available"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def format_rating(value) -> str:
    if pd.isna(value):
        return "Not available"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "Not available"


def get_episode_category(row: pd.Series) -> str:
    if "episode_category" in row and pd.notna(row["episode_category"]):
        return str(row["episode_category"])

    episodes_value = row.get("episodes_numeric", np.nan)
    if pd.isna(episodes_value):
        return "Unknown episodes"

    try:
        episodes_value = float(episodes_value)
    except (TypeError, ValueError):
        return "Unknown episodes"

    if episodes_value == 1:
        return "Single episode"
    if 2 <= episodes_value <= 12:
        return "Short series"
    if 13 <= episodes_value <= 26:
        return "Medium series"
    if 27 <= episodes_value <= 52:
        return "Long series"
    return "Very long series"


def split_genres(genre_value: str) -> list[str]:
    if pd.isna(genre_value):
        return []
    return [
        genre.strip()
        for genre in str(genre_value).split(",")
        if genre.strip() and genre.strip().lower() != "unknown"
    ]


def render_metric_card(label: str, value: str, help_text: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_text(label)}</div>
            <div class="metric-value">{safe_text(value)}</div>
            <div class="metric-help">{safe_text(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_insight(text: str):
    st.markdown(
        f"""
        <div class="insight-box">
            <strong>Data insight:</strong> {safe_text(text)}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_selected_anime_card(row: pd.Series):
    genre_tags = "".join(
        f'<span class="tag">{safe_text(genre)}</span>'
        for genre in split_genres(row.get("genre", ""))[:8]
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="anime-title">🎯 Your selected anime: {safe_text(row.get("anime_name"))}</div>
            <div style="margin-top:0.45rem;">{genre_tags}</div>
            <div class="soft-text" style="margin-top:0.75rem;">
                <strong>Type:</strong> {safe_text(row.get("anime_type"))} &nbsp;|&nbsp;
                <strong>Episodes:</strong> {safe_text(row.get("episodes"))} &nbsp;|&nbsp;
                <strong>Episode profile:</strong> {safe_text(get_episode_category(row))}
                <br>
                <strong>Anime rating:</strong> {format_rating(row.get("anime_rating"))} &nbsp;|&nbsp;
                <strong>Average user rating:</strong> {format_rating(row.get("avg_user_rating"))} &nbsp;|&nbsp;
                <strong>Members:</strong> {format_number(row.get("member_count"))} &nbsp;|&nbsp;
                <strong>Valid user ratings:</strong> {format_number(row.get("valid_rating_count"))}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_recommendation_card(rank: int, row: pd.Series):
    genre_tags = "".join(
        f'<span class="tag">{safe_text(genre)}</span>'
        for genre in split_genres(row.get("genre", ""))[:8]
    )

    st.markdown(
        f"""
        <div class="recommendation-card">
            <div class="anime-title">#{rank} — {safe_text(row.get("anime_name"))}</div>
            <div>
                <span class="score-tag">Similarity: {float(row.get("similarity_score", 0)):.3f}</span>
                <span class="tag">{safe_text(row.get("anime_type"))}</span>
                <span class="tag">{safe_text(get_episode_category(row))}</span>
            </div>
            <div style="margin-top:0.45rem;">{genre_tags}</div>
            <div class="soft-text" style="margin-top:0.75rem;">
                <strong>Anime rating:</strong> {format_rating(row.get("anime_rating"))} &nbsp;|&nbsp;
                <strong>Avg user rating:</strong> {format_rating(row.get("avg_user_rating"))} &nbsp;|&nbsp;
                <strong>Members:</strong> {format_number(row.get("member_count"))} &nbsp;|&nbsp;
                <strong>Valid ratings:</strong> {format_number(row.get("valid_rating_count"))}
                <br>
                Recommended because it has a similar content profile based on genre, anime type, and episode pattern.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def ensure_dashboard_columns(anime_df: pd.DataFrame) -> pd.DataFrame:
    df = anime_df.copy()

    if "anime_name" in df.columns:
        df["anime_name"] = df["anime_name"].apply(lambda x: html_lib.unescape(str(x)))

    if "episode_category" not in df.columns:
        df["episode_category"] = df.apply(get_episode_category, axis=1)

    numeric_columns = [
        "anime_rating",
        "member_count",
        "avg_user_rating",
        "valid_rating_count",
        "total_interactions",
        "unrated_count",
        "episodes_numeric"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.reset_index(drop=True)


@st.cache_resource(show_spinner="Loading recommendation artifacts...")
def load_artifacts():
    required_files = {
        "Anime app data": ANIME_APP_DATA_PATH,
        "TF-IDF matrix": TFIDF_MATRIX_PATH,
    }

    missing_files = [f"{name}: {path}" for name, path in required_files.items() if not path.exists()]
    if missing_files:
        return None, None, None, missing_files

    try:
        anime_df = load_artifact_file(ANIME_APP_DATA_PATH)
        tfidf_matrix = load_artifact_file(TFIDF_MATRIX_PATH)
        metadata = load_json_file(METADATA_PATH)
    except Exception as error:
        return None, None, None, [str(error)]

    if not isinstance(anime_df, pd.DataFrame):
        return None, None, None, ["anime_app_data.pkl did not load as a pandas DataFrame."]

    anime_df = ensure_dashboard_columns(anime_df)

    if len(anime_df) != tfidf_matrix.shape[0]:
        return None, None, None, [
            f"Artifact mismatch: anime_app_data has {len(anime_df)} rows, "
            f"but TF-IDF matrix has {tfidf_matrix.shape[0]} rows."
        ]

    anime_df["selector_label"] = anime_df.apply(
        lambda row: (
            f"{row['anime_name']}  |  "
            f"{row.get('anime_type', 'Unknown')}  |  "
            f"ID: {row.get('anime_id', 'NA')}"
        ),
        axis=1
    )

    return anime_df, tfidf_matrix, metadata, []


def recommend_anime(
    anime_df: pd.DataFrame,
    tfidf_matrix,
    selected_index: int,
    top_n: int = 10,
    anime_type_filter: str = "All types",
    min_rating: float = 0.0
) -> pd.DataFrame:
    similarity_scores = cosine_similarity(tfidf_matrix[selected_index], tfidf_matrix).flatten()

    recommendations = anime_df.copy()
    recommendations["similarity_score"] = similarity_scores

    selected_anime_id = anime_df.iloc[selected_index].get("anime_id")
    recommendations = recommendations[recommendations["anime_id"] != selected_anime_id].copy()

    if anime_type_filter != "All types":
        recommendations = recommendations[recommendations["anime_type"] == anime_type_filter].copy()

    if "anime_rating" in recommendations.columns:
        recommendations = recommendations[
            recommendations["anime_rating"].isna() |
            (recommendations["anime_rating"] >= float(min_rating))
        ].copy()

    sort_columns = ["similarity_score"]
    ascending_values = [False]

    if "anime_rating" in recommendations.columns:
        sort_columns.append("anime_rating")
        ascending_values.append(False)

    if "member_count" in recommendations.columns:
        sort_columns.append("member_count")
        ascending_values.append(False)

    recommendations = recommendations.sort_values(
        by=sort_columns,
        ascending=ascending_values
    ).head(top_n)

    return recommendations.reset_index(drop=True)


def build_genre_frequency(anime_df: pd.DataFrame) -> pd.DataFrame:
    all_genres = []
    for genres in anime_df["genre"].dropna():
        all_genres.extend(split_genres(genres))

    if not all_genres:
        return pd.DataFrame(columns=["genre", "count"])

    return (
        pd.Series(all_genres)
        .value_counts()
        .head(10)
        .reset_index()
        .rename(columns={"index": "genre", 0: "count"})
    )


def simple_bar_chart(data: pd.DataFrame, x_col: str, y_col: str):
    chart_data = data.set_index(x_col)[y_col]
    st.bar_chart(chart_data, use_container_width=True)


# ------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------
anime_df, tfidf_matrix, metadata, load_errors = load_artifacts()

if load_errors:
    st.error("The app could not load the required project artifacts.")
    for error in load_errors:
        st.write(f"- {error}")

    st.info(
        "Please run the notebook from top to bottom so the artifacts are regenerated, "
        "then confirm the files are inside the artifacts folder."
    )
    st.stop()


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
if BANNER_PATH.exists():
    st.image(str(BANNER_PATH), use_container_width=True)
else:
    st.warning(f"Banner image not found at: {BANNER_PATH}")

st.markdown(
    """
    <div class="project-intro">
        <h1>Anime Content-Based Recommendation System</h1>
        <p>
            Discover similar anime using metadata-driven content signals:
            genre, anime type, and episode profile. Ratings and popularity are shown
            for interpretation, not used as the core similarity engine.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
recommend_tab, dashboard_tab = st.tabs(["🎯 Recommend Anime", "📊 Dataset Dashboard"])


# ------------------------------------------------------------
# Tab 1: Recommend Anime
# ------------------------------------------------------------
with recommend_tab:
    st.subheader("Find anime similar to the one you like")

    st.markdown(
        """
        <div class="model-note">
            <strong>How this works:</strong> This app recommends anime by comparing metadata-based content profiles created from genre, anime type, and episode category. 
            These profiles are converted into TF-IDF vectors, and cosine similarity is used to identify anime with the closest content pattern. Ratings and popularity are shown only to help interpret the results; they are not used as the core similarity engine.
        </div>
        """,
        unsafe_allow_html=True
    )

    control_col, profile_col = st.columns([1, 1.15], gap="large")

    with control_col:

        selector_df = anime_df.sort_values("anime_name").copy()
        selector_options = selector_df["selector_label"].tolist()

        preferred_default_titles = [
            "Kimi no Na wa.",
            "Sen to Chihiro no Kamikakushi",
            "Mononoke Hime",
            "Howl no Ugoku Shiro",
            "Byousoku 5 Centimeter"
        ]

        label_by_normalized_title = {
            normalize_title(row["anime_name"]): row["selector_label"]
            for _, row in selector_df.iterrows()
        }

        default_selector_index = 0

        for default_title in preferred_default_titles:
            default_label = label_by_normalized_title.get(normalize_title(default_title))
            if default_label in selector_options:
                default_selector_index = selector_options.index(default_label)
                break

        selected_label = st.selectbox(
            "Search and select an anime",
            options=selector_options,
            index=default_selector_index,
            help="Start typing the anime name to search. The default is a popular anime movie with strong recommendation matches."
        )   

        label_to_index = dict(zip(anime_df["selector_label"], anime_df.index))
        selected_index = int(label_to_index[selected_label])

        top_n = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=15,
            value=5,
            step=1
        )

        type_options = ["All types"] + sorted(
            [str(value) for value in anime_df["anime_type"].dropna().unique()]
        )
        anime_type_filter = st.selectbox(
            "Optional filter: recommendation type",
            options=type_options,
            index=0
        )

        min_rating = st.slider(
            "Optional filter: minimum anime rating",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="This filters the final recommendation list only. It does not change the core similarity model."
        )

        recommend_button = st.button("Recommend Similar Anime", type="primary")

    with profile_col:
        selected_row = anime_df.iloc[selected_index]
        render_selected_anime_card(selected_row)

        st.markdown(
            """
            <div class="info-card">
                <div class="anime-title">Why these inputs matter</div>
                <div class="soft-text">
                    Genre is the strongest content signal. Anime type helps distinguish TV series,
                    movies, OVAs, and specials. Episode profile helps avoid comparing a single movie
                    too aggressively with a very long-running series.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if recommend_button:
        recommendations = recommend_anime(
            anime_df=anime_df,
            tfidf_matrix=tfidf_matrix,
            selected_index=selected_index,
            top_n=top_n,
            anime_type_filter=anime_type_filter,
            min_rating=min_rating
        )

        st.markdown("---")
        st.subheader("Recommended for you")

        if recommendations.empty:
            st.warning(
                "No recommendations matched the selected filters. "
                "Try lowering the minimum rating or choosing 'All types'."
            )
        else:
            for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
                render_recommendation_card(rank, row)

            with st.expander("View recommendations as a table"):
                table_columns = [
                    "anime_name",
                    "similarity_score",
                    "genre",
                    "anime_type",
                    "episode_category",
                    "anime_rating",
                    "member_count",
                    "avg_user_rating",
                    "valid_rating_count"
                ]
                available_columns = [col for col in table_columns if col in recommendations.columns]
                table_df = recommendations[available_columns].copy()

                if "similarity_score" in table_df.columns:
                    table_df["similarity_score"] = table_df["similarity_score"].round(3)

                st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.info("Select an anime and click **Recommend Similar Anime** to generate recommendations.")


# ------------------------------------------------------------
# Tab 2: Dataset Dashboard
# ------------------------------------------------------------
with dashboard_tab:
    st.subheader("Dataset overview and decision insights")

    total_titles = len(anime_df)
    unique_genres = len(set(
        genre
        for genre_list in anime_df["genre"].dropna().apply(split_genres)
        for genre in genre_list
    ))
    avg_anime_rating = anime_df["anime_rating"].mean(skipna=True) if "anime_rating" in anime_df.columns else np.nan
    total_members = anime_df["member_count"].sum(skipna=True) if "member_count" in anime_df.columns else np.nan

    card1, card2, card3, card4 = st.columns(4)
    with card1:
        render_metric_card("Total anime titles", format_number(total_titles), "Rows available for recommendation")
    with card2:
        render_metric_card("Unique genre tags", format_number(unique_genres), "Content signals used for similarity")
    with card3:
        render_metric_card("Average anime rating", format_rating(avg_anime_rating), "Dataset-level quality indicator")
    with card4:
        render_metric_card("Total member count", format_number(total_members), "Popularity reach across titles")

    st.markdown("---")

    chart_col_1, chart_col_2 = st.columns(2, gap="large")

    with chart_col_1:
        st.markdown("### Anime type distribution")
        type_counts = (
            anime_df["anime_type"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )
        type_counts.columns = ["anime_type", "count"]
        simple_bar_chart(type_counts, "anime_type", "count")
        top_type = type_counts.iloc[0]["anime_type"] if not type_counts.empty else "Unknown"
        render_insight(
            f"{top_type} titles dominate the dataset, so the recommender has richer comparison options "
            "for common formats than for rare formats."
        )

    with chart_col_2:
        st.markdown("### Episode category distribution")
        episode_counts = (
            anime_df["episode_category"]
            .fillna("Unknown episodes")
            .value_counts()
            .reset_index()
        )
        episode_counts.columns = ["episode_category", "count"]
        simple_bar_chart(episode_counts, "episode_category", "count")
        render_insight(
            "Episode profile helps the app compare titles more fairly, especially when separating "
            "single-episode movies from long-running series."
        )

    chart_col_3, chart_col_4 = st.columns(2, gap="large")

    with chart_col_3:
        st.markdown("### Top 10 genre tags")
        genre_freq = build_genre_frequency(anime_df)
        simple_bar_chart(genre_freq, "genre", "count")
        render_insight(
            "Frequently occurring genres such as action, comedy, fantasy, romance, and sci-fi become "
            "major content drivers in the recommendation engine."
        )

    with chart_col_4:
        st.markdown("### Anime rating distribution")
        rating_data = anime_df[["anime_rating"]].dropna().copy()
        if rating_data.empty:
            st.warning("Anime rating data is not available.")
        else:
            bins = pd.cut(
                rating_data["anime_rating"],
                bins=[0, 4, 5, 6, 7, 8, 9, 10],
                include_lowest=True
            )
            rating_counts = bins.value_counts().sort_index().reset_index()
            rating_counts.columns = ["rating_range", "count"]
            rating_counts["rating_range"] = rating_counts["rating_range"].astype(str)
            simple_bar_chart(rating_counts, "rating_range", "count")
        render_insight(
            "Most anime ratings are concentrated in the mid-to-high range, so rating is better used "
            "as supporting information rather than the main similarity signal."
        )

    st.markdown("### Popularity and engagement")

    chart_col_5, chart_col_6 = st.columns(2, gap="large")

    with chart_col_5:
        st.markdown("#### Top 10 anime by member count")
        if "member_count" in anime_df.columns:
            top_members = (
                anime_df.sort_values("member_count", ascending=False)
                .head(10)[["anime_name", "member_count"]]
                .copy()
            )
            simple_bar_chart(top_members, "anime_name", "member_count")
        else:
            st.warning("Member count column is not available.")

        render_insight(
            "High member count shows audience reach and popularity, but popularity is not the same as "
            "content similarity. The app shows it for decision support only."
        )

    with chart_col_6:
        st.markdown("#### Top 10 anime by valid rating count")
        if "valid_rating_count" in anime_df.columns:
            top_valid_ratings = (
                anime_df.sort_values("valid_rating_count", ascending=False)
                .head(10)[["anime_name", "valid_rating_count"]]
                .copy()
            )
            simple_bar_chart(top_valid_ratings, "anime_name", "valid_rating_count")
        else:
            st.warning("Valid rating count column is not available.")

        render_insight(
            "Titles with many valid ratings have stronger audience feedback, making their rating statistics "
            "more reliable than anime with only a small number of ratings."
        )

    with st.expander("View dataset sample used by the app"):
        preview_columns = [
            "anime_id",
            "anime_name",
            "genre",
            "anime_type",
            "episode_category",
            "anime_rating",
            "member_count",
            "avg_user_rating",
            "valid_rating_count"
        ]
        available_preview_columns = [col for col in preview_columns if col in anime_df.columns]
        st.dataframe(
            anime_df[available_preview_columns].head(25),
            use_container_width=True,
            hide_index=True
        )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown(
    """
    <div class="model-note">
        <strong>Project note:</strong> This is a content-based recommender. It does not use user-user
        collaborative filtering or matrix factorization. The recommendation logic is intentionally aligned
        with the assignment requirement: recommend similar anime using anime content metadata.
    </div>
    """,
    unsafe_allow_html=True
)
