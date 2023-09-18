from fastapi import FastAPI
from typing import List

from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String

from datetime import datetime

import os
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
from catboost import CatBoostClassifier

from schema import PostGet

if __name__ == '__main__':
    load_dotenv()


app = FastAPI()

SQLALCHEMY_DATABASE_URL = os.getenv('DATABASE_URL')

engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=100, max_overflow=100)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)


Base = declarative_base()


def get_db():
    with SessionLocal() as db:
        return db

def load_models():
    model_path = path
    model = CatBoostClassifier()
    model.load_model(model_path, format='cbm')

    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features() -> list:
    con = SQLALCHEMY_DATABASE_URL

    logger.info("loading already liked posts")
    liked_posts_query = """
        SELECT DISTINCT post_id, user_id
        FROM feed_data
        WHERE action = 'like'
    """

    liked_posts = batch_load_sql(liked_posts_query)

    logger.info('loading posts features')
    posts_features = pd.read_sql('artyom_pavlov_posts_features', con=con)

    logger.info('loading user features')
    users_features = pd.read_sql('artyom_pavlov_users_features', con=con)

    return [liked_posts, posts_features, users_features]


logger.info('loading model')
path = "models/catboost_git"
model = load_models()

logger.info('loading features')
features = load_features()

logger.info('service is up and running')


def get_recommends(id: int, time: datetime, limit: int):
    # Загрузка фичей пользователя
    logger.info(f"user_id: {id}")
    logger.info("loading user's features")
    user_features = features[2].loc[features[2].user_id == id]
    user_features.drop(['user_id', 'index'], axis=1, inplace=True)

    # Загрузка фичей постов
    logger.info('loading posts')
    posts_features = features[1].drop(['index', 'text'], axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    # Объединение фичей
    logger.info('merging features')
    add_user = dict(zip(user_features.columns, user_features.values[0]))

    all_features = posts_features.assign(**add_user)
    all_features = all_features.set_index('post_id')

    logger.info("adding time info")
    all_features['hour'] = time.hour
    all_features['month'] = time.month
    all_features['day_of_week'] = time.weekday()

    logger.info('predicting...')
    predicts = model.predict_proba(all_features)[:, 1]
    all_features['pred_probas'] = predicts

    logger.info("filtering posts")
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = all_features[~all_features.index.isin(liked_posts)]

    recommended_posts = filtered_.sort_values('pred_probas', ascending=False)[: limit].index

    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts
    ]


@app.get("/post/recommendations/", response_model=List[PostGet])
def top_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    return get_recommends(id, time, limit)



