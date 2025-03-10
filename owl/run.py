from dotenv import load_dotenv
load_dotenv()

from camel.models import ModelFactory
from camel.toolkits import *
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

from typing import List, Dict

from retry import retry
from loguru import logger

from utils import OwlRolePlaying, run_society
import os




def construct_society(question: str) -> OwlRolePlaying:
    r"""Construct the society based on the question."""

    user_role_name = "user"
    assistant_role_name = "assistant"
    
    user_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(), # [Optional] the config for model
    )

    assistant_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(), # [Optional] the config for model
    )

    tools_list = [
        *WebToolkit(
            headless=False, 
            web_agent_model=assistant_model, 
            planning_agent_model=assistant_model
        ).get_tools(),
        *DocumentProcessingToolkit().get_tools(),
        *VideoAnalysisToolkit(model=assistant_model).get_tools(),  # This requires OpenAI Key
        *AudioAnalysisToolkit().get_tools(),  # This requires OpenAI Key
        *CodeExecutionToolkit().get_tools(),
        #*ImageAnalysisToolkit(model=assistant_model).get_tools(),
        #*SearchToolkit(model=assistant_model).get_tools(),
        #*ExcelToolkit().get_tools()
    ]

    user_role_name = 'user'
    user_agent_kwargs = dict(model=user_model)
    assistant_role_name = 'assistant'
    assistant_agent_kwargs = dict(model=assistant_model,
    tools=tools_list)
    
    task_kwargs = {
        'task_prompt': question,
        'with_task_specify': False,
    }

    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name=user_role_name,
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name=assistant_role_name,
        assistant_agent_kwargs=assistant_agent_kwargs,
    )
    
    return society


# Example case
question = "Why the world keep running out of water?"

society = construct_society(question)
answer, chat_history, token_count = run_society(society)

logger.success(f"Answer: {answer}")





