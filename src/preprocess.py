import pandas as pd
import streamlit as st
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

from src.text_utils import split_into_sentences

# Load environment variables from .env file
load_dotenv()


def load_data():
    engine_dws = create_engine(
        "postgresql://{}:{}@{}:{}/dws".format(
            os.environ.get("DWS_USER"),
            os.environ.get("DWS_PWD"),
            os.environ.get("DWS_HOST"),
            os.environ.get("DWS_PORT"),
        )
    )

    # Execute the SQL query using SQLAlchemy
    query = """
    select *
    from dm_smarthome_analytics_team.csi_source cs
    where cs.comment_text is not null
        and main_score < 5
        and cs.question not in ('Откуда вы узнали об умном доме Sber?')
    limit 10
    """
    with engine_dws.connect() as connection:
        data = pd.read_sql(query, connection)

    print(f"got data from SQL: {data.head()}")

    return data


def load_data_file():
    # Load data from the specified Excel file and sheet
    return pd.read_excel("data/input.xlsx", sheet_name="массив_full")


def rename_columns(df):

    column_mapping = {
        "ROW_ID": "row_id",
        "EVENT_TYPE_ID": "event_type_id",
        "EVENT_NAME": "event_name",
        "product_name": "product_name",
        "QUESTION_NUM": "question_num",
        "QUESTION": "question",
        "COMMENT_TEXT": "comment",
        "ANSWER_TXT": "answer",
        "ANSWER_DT": "datetime",
        "segm_general": "segment",
        "v21": "v21",
        "v22": "v22",
        "v23": "v23",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "Месяц": "month",
        "Неделя": "week",
        "квартал": "quarter",
    }
    return df.rename(columns=column_mapping)


def parse_dates(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(
            df[col], utc=True, errors="coerce"
        )  # Use errors='coerce' to handle invalid parsing
        # Convert to date, handling NaT values by converting them to None
        df[col] = df[col].apply(lambda x: x if pd.notna(x) else None)

    return df


def extract_comments(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Keep only rows with non-empty 'comment'
    df = df[df["comment"].notna() & (df["comment"] != "")]

    # Filter for rows where 'datetime' is in January 2025
    df = df[df["datetime"].dt.year == 2025]
    df = df[df["datetime"].dt.month == 1]

    # print(df.info())

    return df


@st.cache_data(show_spinner=False)
def explode_reviews(df, column_name):
    """
    A function that explodes the reviews with multiple
    sentences into multiple rows with 1 sentence each.

    Parameters:
    - df: A pandas DataFrame. The DataFrame containing the reviews.
    - column_name: A string. The name of the column containing the reviews.

    Returns:
    - df: A pandas DataFrame. The DataFrame with exploded sentences.
    """
    df = df.copy()
    # Split reviews into sentences
    df[column_name] = df[column_name].astype(str).apply(split_into_sentences)

    # Explode the DataFrame and reset the index
    return df.explode(column_name).reset_index(drop=True).dropna(subset=[column_name])


def preprocess_data():
    data = load_data_file()
    df = rename_columns(data)
    df = parse_dates(df, ["datetime"])
    df = extract_comments(df)
    return df
