import os
import sys
import logging
from langchain import hub
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Constants
MODEL = "gpt-4"
TEMPERATURE = 0

# Environment variable for OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set.")
    raise EnvironmentError("OPENAI_API_KEY environment variable is required.")

# Prompt from LangChain hub
PROMPT = hub.pull("rlm/rag-prompt")

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
