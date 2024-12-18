from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongo_uri: SecretStr
    openai_key: SecretStr
    tavily_api_key: SecretStr

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_project: str = "ai-agent"
    langchain_api_key: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
