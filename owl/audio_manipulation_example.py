#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audio Manipulation Toolkit Example
This script demonstrates how to use the AudioManipulationToolkit with the OWL framework.
"""
import sys

import os

# Add the parent directory and camel directory to the Python path to find the local CAMEL module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "camel"))

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from camel.toolkits import AudioManipulationToolkit
from loguru import logger

from utils import OwlRolePlaying, run_society


def construct_society_with_audio_tools(question: str) -> OwlRolePlaying:
    """Construct an OWL society with audio manipulation tools.
    
    Args:
        question (str): The task or question to be addressed.
        
    Returns:
        OwlRolePlaying: The configured society with audio tools.
    """
    # Define role names
    user_role_name = "Audio Producer"
    assistant_role_name = "Sound Engineer"
    
    # Create models for the agents
    user_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(),
    )

    assistant_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict=ChatGPTConfig(temperature=0.2, top_p=1).as_dict(),
    )

    # Create the Audio Manipulation toolkit
    audio_toolkit = AudioManipulationToolkit(
        cache_dir="./output/audio",
        audio_ldm_model="cvssp/audioldm-s-full-v2",
        tts_model="facebook/fastspeech2-en-ljspeech"
    )
    
    # Get all the audio tools
    audio_tools = audio_toolkit.get_tools()
    
    # Configure the agents
    user_agent_kwargs = dict(model=user_model)
    assistant_agent_kwargs = dict(
        model=assistant_model,
        tools=audio_tools
    )
    
    # Configure the task
    task_kwargs = {
        'task_prompt': question,
        'with_task_specify': False,
    }

    # Create the society
    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name=user_role_name,
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name=assistant_role_name,
        assistant_agent_kwargs=assistant_agent_kwargs,
    )
    
    return society


def run_audio_task(task_description: str) -> str:
    """Run an audio task using the AudioManipulationToolkit.
    
    Args:
        task_description (str): Description of the audio task to perform.
        
    Returns:
        str: Path to the resulting audio file.
    """
    logger.info(f"Starting audio task: {task_description}")
    
    # Create the society with audio tools
    society = construct_society_with_audio_tools(task_description)
    
    # Run the society to complete the task
    answer, chat_history, token_count = run_society(society)
    
    logger.success(f"Task completed with {token_count} tokens used")
    logger.info(f"Answer: {answer}")
    
    # Extract the path to the resulting audio file from the answer
    audio_path = None
    for line in answer.split("\n"):
        if "output_audio_path" in line or "audio_path" in line:
            parts = line.split(":", 1)
            if len(parts) > 1:
                candidate = parts[1].strip().strip("'\"")
                if os.path.exists(candidate) and candidate.endswith((".wav", ".mp3")):
                    audio_path = candidate
                    break
    
    if not audio_path:
        logger.warning("Could not find audio path in the answer. Check the chat history for details.")
    
    return audio_path or answer


def direct_toolkit_example():
    """Example of directly using the AudioManipulationToolkit without OWL society."""
    # Create output directory
    os.makedirs("./output/audio/direct", exist_ok=True)
    
    # Initialize the toolkit
    audio_toolkit = AudioManipulationToolkit(
        cache_dir="./output/audio/direct"
    )
    
    try:
        # Example 1: Generate environmental sounds
        forest_audio_path = audio_toolkit.generate_audio(
            instruction="A peaceful forest with birds chirping and a stream flowing",
            output_audio_path="./output/audio/direct/forest_scene.wav",
            duration=10.0
        )
        logger.success(f"Generated forest audio: {forest_audio_path}")
        
        # Example 2: Generate speech
        speech_audio_path = audio_toolkit.generate_audio(
            instruction="Say 'Welcome to the audio manipulation demo'",
            output_audio_path="./output/audio/direct/welcome_speech.wav"
        )
        logger.success(f"Generated speech audio: {speech_audio_path}")
        
        # Example 3: Edit audio
        edited_audio_path = audio_toolkit.edit_audio(
            input_audio_path=forest_audio_path,
            instruction="Add the sound of thunder in the background",
            output_audio_path="./output/audio/direct/forest_with_thunder.wav"
        )
        logger.success(f"Edited audio: {edited_audio_path}")
        
        # Example 4: Combine audio files
        combined_audio_path = audio_toolkit.combine_audio_files(
            audio_paths=[speech_audio_path, forest_audio_path],
            output_audio_path="./output/audio/direct/combined.wav",
            instructions="Crossfade 500 milliseconds between the two audio files"
        )
        logger.success(f"Combined audio: {combined_audio_path}")
        
        # Example 5: Create an audio script
        script_result = audio_toolkit.create_audio_script(
            script_prompt="Create a short radio advertisement for a camping gear store",
            output_audio_path="./output/audio/direct/camping_ad.wav",
            output_script_path="./output/audio/direct/camping_ad.txt"
        )
        logger.success(f"Created audio script: {script_result['script_path']}")
        logger.success(f"Created audio for script: {script_result['audio_path']}")
        
        # Example 6: Check watermark
        watermark_result = audio_toolkit.check_watermark(forest_audio_path)
        logger.info(f"Watermark check result: {watermark_result}")
        
    except Exception as e:
        logger.error(f"Error in direct toolkit example: {e}")


if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs("./output/audio", exist_ok=True)
    
    # Choose which example to run
    # 1. Run through OWL society
    # 2. Direct toolkit usage
    
    example_choice = 1  # Change to 1 to use the OWL society
    
    if example_choice == 1:
        # Example 1: Generate environmental audio
        task1 = """
        Create a 10-second audio clip of a serene forest scene with a gentle 
        stream, birds singing, and a light breeze rustling the leaves.
        """
        
        # Example 2: Generate speech
        task2 = """
        Generate speech that says "Welcome to the audio manipulation demo. 
        This toolkit allows you to create and edit audio using natural language instructions."
        """
        
        # Example 3: Create a podcast intro
        task3 = """
        Create a short podcast intro for a show called "Tech Horizons" that 
        discusses emerging technologies. The intro should be energetic and 
        include some subtle electronic background music.
        """
        
        # Run one of the example tasks
        result = run_audio_task(task2)
        logger.info(f"The resulting audio is available at: {result}")
    else:
        # Run the direct toolkit examples
        direct_toolkit_example() 