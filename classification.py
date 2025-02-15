# %%[markdown]
# # Использование Гигачата для классификации комментариев
# %%[markdown]
# ## Giga docs
# %%[markdown]
# Initialization
# %%
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from enum import Enum
from typing import List
from dotenv import load_dotenv
import os
import requests
from tqdm import tqdm

import pandas as pd
import warnings

import httpx
import logging
import socket
import ssl
from requests.exceptions import RequestException
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# %%

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# %%
df = pd.read_excel("data/input.xlsx", sheet_name="массив_full")

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

df = df.rename(columns=column_mapping)


df["datetime"] = pd.to_datetime(
    df["datetime"], utc=True, errors="coerce"
)  # Use errors='coerce' to handle invalid parsing
# Convert to date, handling NaT values by converting them to None
df["datetime"] = df["datetime"].apply(lambda x: x if pd.notna(x) else None)

# %%
# Keep only rows with non-empty 'comment'
df = df[df["comment"].notna() & (df["comment"] != "")]

# Drop duplicates
df = df.drop_duplicates(subset=["comment"])

# Filter for rows where 'datetime' is in January 2025
df = df[df["datetime"].dt.year == 2025]
df = df[df["datetime"].dt.month == 1]

df = df[df["answer"] != 5]

# %%
df = df.head(10)
print(df.info())
# %%
# --------------------------------------------------------------
# using OpenAI
# --------------------------------------------------------------

# https://developers.sber.ru/docs/ru/gigachat/api/compatible-openai
# Load environment variables from .env file
load_dotenv()

# Retrieve GIGACHAT_CLIENT_ID and auth_key from environment variables
GIGACHAT_CLIENT_ID = os.getenv("GIGACHAT_CLIENT_ID")
auth_key = os.getenv("GIGACHAT_AUTH_KEY")

payload = {"scope": "GIGACHAT_API_PERS"}
headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "application/json",
    "RqUID": GIGACHAT_CLIENT_ID,
    "Authorization": "Basic " + auth_key,
}


# Define the Config class with the auth_url
class Config:
    auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"


# Path to your certificate file
cert_path = "/Users/denis/Documents/russiantrustedca/russiantrustedca.pem"

response = requests.request(
    "POST", Config.auth_url, headers=headers, data=payload, verify=cert_path
)

data = response.json()
access_token = data.get("access_token", "Unknown")


# %%
# Add this function to test connectivity
def test_connection():
    try:
        # Test basic internet connectivity
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        logger.info("Internet connection: OK")

        # Test HTTPS connectivity to Sber API
        response = requests.get(
            "https://gigachat.devices.sberbank.ru", verify=cert_path
        )
        logger.info(f"Sber API connection: OK (Status: {response.status_code})")

        return True
    except socket.error as e:
        logger.error(f"Internet connection failed: {str(e)}")
    except RequestException as e:
        logger.error(f"Sber API connection failed: {str(e)}")
    except ssl.SSLError as e:
        logger.error(f"SSL Certificate error: {str(e)}")
    except Exception as e:
        logger.error(f"Other error: {str(e)}")
    return False


# Modify the authentication request with better error handling
try:
    logger.info("Testing connection before authentication...")
    if not test_connection():
        raise Exception("Connection test failed")

    logger.info("Attempting authentication...")
    response = requests.request(
        "POST",
        Config.auth_url,
        headers=headers,
        data=payload,
        verify=cert_path,
        timeout=10,  # Add timeout
    )
    response.raise_for_status()  # Will raise an exception for 4XX/5XX status codes

    data = response.json()
    access_token = data.get("access_token")
    if not access_token:
        raise ValueError("No access token in response")

    logger.info("Authentication successful")
except Exception as e:
    logger.error(f"Authentication failed: {str(e)}")
    logger.error(f"Response status code: {getattr(response, 'status_code', 'N/A')}")
    logger.error(f"Response content: {getattr(response, 'text', 'N/A')}")
    raise

# %%[markdown]

# --------------------------------------------------------------
# Step 1: Get clear on your objectives
# --------------------------------------------------------------

"""
### Objective: Develop an AI-powered comments classification system that:
- Accurately categorizes comments
- Assesses the sentiment of each ticket
- Extracts key information for quick resolution
- Provides confidence scores to flag uncertain cases for human review

### Business impact:
- Improve customer satisfaction by prioritizing negative sentiment comments
- Increase efficiency by providing agents with key information upfront
- Optimize workforce allocation by automating routine classifications
"""
# %%
# --------------------------------------------------------------
# Step 2: Patch your LLM with instructor
# --------------------------------------------------------------

# Instructor makes it easy to get structured data like JSON from LLMs
client = instructor.patch(
    OpenAI(
        api_key=access_token,
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
        http_client=httpx.Client(verify=cert_path),
    ),
    mode=instructor.Mode.JSON,
)

# %%

# --------------------------------------------------------------
# Step 3: Define Pydantic data models
# --------------------------------------------------------------

"""
This code defines a structured data model for classifying customer comments
using Pydantic and Python's Enum class.
It specifies categories, urgency levels, customer sentiments, and other
relevant information as predefined options or constrained fields.
This structure ensures data consistency, enables automatic validation,
and facilitates easy integration with AI models and other parts of a system.
"""


class TicketCategory(str, Enum):
    ADVERTISING = "проблемы с рекламой"
    SLOW_PERFORMANCE = "медленная работа, зависания"
    SUBSCRIPTIONS = "проблемы с управлением подписками"
    NETWORK_ISSUE = "проблемы с интернетом, Wi-Fi"
    ASSISTANT_ISSUE = "проблемы с голосовым управлением"
    UX_ISSUE = "проблемы с интерфейсом"
    PULT_ISSUE = "проблемы с пультом управления"
    IMAGE_ISSUE = "проблемы с качеством изображения"
    DOWNLOADS_ISSUE = "проблемы с загрузкой приложений"
    SOUND_ISSUE = "проблемы с звуком"
    OTHER = "прочие"


class CustomerSentiment(str, Enum):
    ANGRY = "злой"
    FRUSTRATED = "недовольный"
    NEUTRAL = "нейтральный"
    SATISFIED = "довольный"


class TicketClassification(BaseModel):
    category: TicketCategory
    sentiment: CustomerSentiment
    confidence: float = Field(
        ge=0, le=1, description="Confidence score for the classification"
    )
    key_information: List[str] = Field(
        description="List of key points extracted from the comment"
    )


# %%

# --------------------------------------------------------------
# Step 5: Optimize your prompts and experiment
# --------------------------------------------------------------
# To optimize:
# 1. Refine the system message to provide more context about your business
# 2. Experiment with different models
# 3. Fine-tune the model on your specific ticket data if available (topic modeling)
# 4. Adjust the TicketClassification model based on business needs

SYSTEM_PROMPT = """
Ты помощник команды разработки продукта компании по производству электроники
и программного обеспечения для мультимедиа и умного дома.
Твоя роль заключается в анализе комментариев клиентов в опросе CSI
(Customer satisfaction index) и предоставлении структурированной информации
для помощи нашей команде в оценке проблем пользователей.

Контекст бизнеса:
- Мы обрабатываем тысячи комментариев в различных категориях
(продукты, технические проблемы, сложности в использовании).
- Быстрая и точная классификация имеет решающее значение
для понимания удовлетворенности клиентов и эффективности работы
над улучшением продуктов.
- Мы приоритизируем на основе количества комментариев связанных
с одной и той же проблемой.

Твои задачи:
1. Категоризуй комментарий в наиболее подходящую категорию.
2. Определи настроение клиента.
3. Извлеки ключевую информацию, которая будет полезна для нашей продуктовой команды.
4. Предоставь оценку уверенности в твоей классификации.

Помни:
- Будь объективным и основывай свой анализ только на информации,
предоставленной в комментарии.
- Если ты не уверен в каком-либо аспекте, отражай это в оценке уверенности.
- Для 'key_information' извлекай конкретные детали, такие как номера заказов,
названия продуктов или проблемы.

Пронализируй комментарий клиента и предоставь запрошенную информацию
в указанном формате.
"""


def classify_ticket(ticket_text: str) -> TicketClassification | None:
    try:
        response = client.chat.completions.create(
            model="GigaChat",
            response_model=TicketClassification,
            temperature=0,
            max_retries=3,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {"role": "user", "content": ticket_text},
            ],
        )
        return response
    except Exception as e:
        print(f"Error processing ticket: {e}")
        return None


# %%
# pandas loop
tqdm.pandas()

# Process comments and store results
df["classified"] = df["comment"].progress_apply(
    lambda x: (
        result.model_dump()
        if (x and isinstance(x, str) and (result := classify_ticket(x)) is not None)
        else None
    )
)

# Parse the JSON response into separate columns and maintain index alignment
df_classified = pd.json_normalize(df["classified"].dropna()).set_index(
    df["classified"].dropna().index
)

# Drop the original classified column to avoid duplication
df = df.drop(columns=["classified"])

# Now concatenate with aligned indexes
df = pd.concat([df, df_classified], axis=1)
df.to_csv("./data/classified.csv", index=False, sep=";")


# %%
print(df[["comment", "category", "sentiment", "confidence", "key_information"]])


# %%

# Create a bar plot of category distribution
plt.figure(figsize=(12, 6))
df["category"].value_counts().plot(kind="bar")
plt.title("Distribution of Feedback Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
