# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import os
import sys
import tempfile
import shutil
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import importlib.util

from loguru import logger

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool


class WavCraftAPI:
    """A wrapper class that interfaces with WavCraft modules directly 
    instead of calling CLI scripts.
    """
    
    def __init__(
        self, 
        wavcraft_path: Optional[str] = None, 
        model: str = "gpt-4o"
    ):
        """Initialize the WavCraft API wrapper.
        
        Args:
            wavcraft_path (str, optional): Path to WavCraft installation.
                If None, uses environment variable WAVCRAFT_PATH.
            model (str, optional): LLM to use for operations.
                Defaults to "gpt-4o".
        """
        self.wavcraft_path = wavcraft_path or os.environ.get("WAVCRAFT_PATH")
        if not self.wavcraft_path:
            raise ValueError(
                "WavCraft path must be provided either directly or through "
                "the WAVCRAFT_PATH environment variable"
            )
            
        self.model = model
        
        # Add WavCraft to Python path for imports
        if self.wavcraft_path not in sys.path:
            sys.path.insert(0, self.wavcraft_path)
        
        # Try to import WavCraft modules
        try:
            # Import core modules
            self.wavcraft_core = self._import_module("wavcraft.core")
            self.wavcraft_utils = self._import_module("wavcraft.utils")
            self.wavcraft_models = self._import_module("wavcraft.models")
            
            # Import specialized modules
            self.audio_editor = self._import_module("wavcraft.editor")
            self.audio_generator = self._import_module("wavcraft.generator")
            self.script_generator = self._import_module("wavcraft.scriptwriter")
            self.watermark = self._import_module("wavcraft.watermark")
            
            self._loaded = True
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f"Failed to import WavCraft modules: {e}")
            self._loaded = False
            raise ImportError(
                f"Could not import WavCraft modules from {self.wavcraft_path}. "
                f"Please ensure WavCraft is properly installed."
            )
    
    def _import_module(self, module_name: str) -> Any:
        """Safely import a module by name.
        
        Args:
            module_name (str): Name of the module to import
            
        Returns:
            Any: The imported module
        
        Raises:
            ImportError: If the module cannot be imported
        """
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            # For modules inside WavCraft that might not be in Python path
            module_spec = importlib.util.find_spec(
                module_name, 
                [self.wavcraft_path]
            )
            if not module_spec:
                raise ImportError(f"Module {module_name} not found")
                
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            return module
    
    def edit_audio(
        self, 
        input_path: str, 
        instruction: str, 
        output_path: str
    ) -> str:
        """Edit audio using WavCraft's editor module.
        
        Args:
            input_path (str): Path to input audio file
            instruction (str): Natural language instruction
            output_path (str): Path to save output audio
            
        Returns:
            str: Path to the edited audio file
        """
        logger.info(f"Editing audio: {input_path} with instruction: {instruction}")
        
        try:
            # Create editor configuration
            config = {
                "model": self.model,
                "input_file": input_path,
                "instruction": instruction,
                "output_file": output_path
            }
            
            # Call the editor API
            result = self.audio_editor.edit_audio(config)
            
            if not result.get("success", False):
                raise RuntimeError(f"Audio editing failed: {result.get('message', 'Unknown error')}")
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error in WavCraft edit_audio: {str(e)}")
            raise
    
    def generate_audio(
        self, 
        instruction: str, 
        output_path: str,
        duration: Optional[float] = None
    ) -> str:
        """Generate audio using WavCraft's generator module.
        
        Args:
            instruction (str): Natural language description of audio to generate
            output_path (str): Path to save generated audio
            duration (float, optional): Approximate duration in seconds
            
        Returns:
            str: Path to the generated audio file
        """
        logger.info(f"Generating audio: {instruction}")
        
        try:
            # Prepare the generation prompt
            if duration:
                generation_prompt = f"Generate a {duration} second audio of {instruction}"
            else:
                generation_prompt = f"Generate audio of {instruction}"
            
            # Create generator configuration
            config = {
                "model": self.model,
                "instruction": generation_prompt,
                "output_file": output_path
            }
            
            # Call the generator API
            result = self.audio_generator.generate_audio(config)
            
            if not result.get("success", False):
                raise RuntimeError(f"Audio generation failed: {result.get('message', 'Unknown error')}")
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error in WavCraft generate_audio: {str(e)}")
            raise
    
    def combine_audio_files(
        self, 
        input_paths: List[str], 
        output_path: str,
        instructions: Optional[str] = None
    ) -> str:
        """Combine multiple audio files with optional editing.
        
        Args:
            input_paths (List[str]): List of paths to audio files to combine
            output_path (str): Path to save combined audio
            instructions (str, optional): Instructions for combining
            
        Returns:
            str: Path to the combined audio file
        """
        logger.info(f"Combining {len(input_paths)} audio files")
        
        try:
            # Prepare configuration
            config = {
                "model": self.model,
                "input_files": input_paths,
                "output_file": output_path
            }
            
            if instructions:
                config["instruction"] = instructions
            
            # Call the editor for combining audio
            result = self.audio_editor.combine_audio(config)
            
            if not result.get("success", False):
                raise RuntimeError(f"Audio combination failed: {result.get('message', 'Unknown error')}")
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error in WavCraft combine_audio_files: {str(e)}")
            raise
    
    def create_audio_script(
        self, 
        prompt: str, 
        output_audio_path: str,
        output_script_path: str
    ) -> Dict[str, str]:
        """Create an audio script and its corresponding audio.
        
        Args:
            prompt (str): Prompt for the script to generate
            output_audio_path (str): Path to save output audio
            output_script_path (str): Path to save output script
            
        Returns:
            Dict[str, str]: Dictionary with paths to audio and script files
        """
        logger.info(f"Creating audio script: {prompt}")
        
        try:
            # Create script generator configuration
            config = {
                "model": self.model,
                "prompt": prompt,
                "audio_output": output_audio_path,
                "script_output": output_script_path
            }
            
            # Call the script generator API
            result = self.script_generator.create_script(config)
            
            if not result.get("success", False):
                raise RuntimeError(f"Script creation failed: {result.get('message', 'Unknown error')}")
                
            return {
                "audio_path": output_audio_path,
                "script_path": output_script_path
            }
            
        except Exception as e:
            logger.error(f"Error in WavCraft create_audio_script: {str(e)}")
            raise
    
    def check_watermark(self, audio_path: str) -> Dict[str, Any]:
        """Check if an audio file contains a WavCraft watermark.
        
        Args:
            audio_path (str): Path to the audio file to check
            
        Returns:
            Dict[str, Any]: Information about the watermark
        """
        logger.info(f"Checking watermark for: {audio_path}")
        
        try:
            # Call the watermark detector API
            result = self.watermark.detect_watermark(audio_path)
            
            watermark_found = result.get("has_watermark", False)
            
            output = {
                "has_watermark": watermark_found,
                "output": result.get("message", "")
            }
            
            if watermark_found and "confidence" in result:
                output["confidence"] = result["confidence"]
                
            return output
            
        except Exception as e:
            logger.error(f"Error in WavCraft check_watermark: {str(e)}")
            return {"has_watermark": False, "error": str(e)}


class WavCraftToolkit(BaseToolkit):
    r"""A toolkit for audio manipulation using WavCraft's internal API.
    
    This toolkit integrates with WavCraft (https://github.com/JinhuaLiang/WavCraft), 
    an AI agent for audio creation and editing using Large Language Models.
    It uses direct module imports instead of subprocess calls.
    """

    def __init__(
        self,
        wavcraft_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model: str = "gpt-4o",
    ):
        """Initialize the WavCraft toolkit.
        
        Args:
            wavcraft_path (str, optional): Path to the WavCraft installation.
                If None, will attempt to find it through WAVCRAFT_PATH env var.
            cache_dir (str, optional): Directory to cache audio files.
                Defaults to 'tmp/'.
            model (str, optional): LLM to use for WavCraft operations.
                Defaults to "gpt-4o".
        """
        self.cache_dir = cache_dir or "tmp/"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize the WavCraft API wrapper
        try:
            self.api = WavCraftAPI(wavcraft_path=wavcraft_path, model=model)
            self.is_available = True
        except (ImportError, ValueError) as e:
            logger.warning(f"WavCraft API initialization failed: {e}")
            self.is_available = False

    def edit_audio(
        self, 
        input_audio_path: str, 
        instruction: str, 
        output_audio_path: Optional[str] = None
    ) -> str:
        """Edit an audio file using natural language instructions.
        
        Args:
            input_audio_path (str): Path to the input audio file.
            instruction (str): Natural language instruction for editing.
            output_audio_path (str, optional): Path to save the output audio.
                If None, a path will be generated in the cache directory.
                
        Returns:
            str: Path to the edited audio file.
        """
        self._check_availability()
        
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"Input audio file not found: {input_audio_path}")
        
        if output_audio_path is None:
            # Generate a filename based on the instruction and input filename
            input_filename = os.path.basename(input_audio_path)
            input_name, input_ext = os.path.splitext(input_filename)
            safe_instruction = "".join(
                c if c.isalnum() else "_" for c in instruction[:20]
            )
            output_filename = f"{input_name}_{safe_instruction}{input_ext}"
            output_audio_path = os.path.join(self.cache_dir, output_filename)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        
        # Call the API to edit the audio
        try:
            result_path = self.api.edit_audio(
                input_path=input_audio_path,
                instruction=instruction,
                output_path=output_audio_path
            )
            
            logger.success(f"Successfully edited audio: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Error editing audio: {str(e)}")
            raise RuntimeError(f"Failed to edit audio with WavCraft: {str(e)}")

    def generate_audio(
        self, 
        instruction: str, 
        output_audio_path: Optional[str] = None,
        duration: Optional[float] = None
    ) -> str:
        """Generate an audio file from a textual description.
        
        Args:
            instruction (str): Description of the audio to generate.
            output_audio_path (str, optional): Path to save the output audio.
                If None, a path will be generated in the cache directory.
            duration (float, optional): Approximate duration of audio in seconds.
                
        Returns:
            str: Path to the generated audio file.
        """
        self._check_availability()
        
        if output_audio_path is None:
            # Generate a filename based on the instruction
            safe_instruction = "".join(
                c if c.isalnum() else "_" for c in instruction[:20]
            )
            output_filename = f"generated_{safe_instruction}.wav"
            output_audio_path = os.path.join(self.cache_dir, output_filename)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        
        # Call the API to generate audio
        try:
            result_path = self.api.generate_audio(
                instruction=instruction,
                output_path=output_audio_path,
                duration=duration
            )
            
            logger.success(f"Successfully generated audio: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise RuntimeError(f"Failed to generate audio with WavCraft: {str(e)}")

    def combine_audio_files(
        self, 
        audio_paths: List[str], 
        output_audio_path: Optional[str] = None,
        instructions: Optional[str] = None
    ) -> str:
        """Combine multiple audio files with optional editing.
        
        Args:
            audio_paths (List[str]): List of paths to audio files to combine.
            output_audio_path (str, optional): Path to save the combined audio.
                If None, a path will be generated in the cache directory.
            instructions (str, optional): Instructions for how to combine the files.
                
        Returns:
            str: Path to the combined audio file.
        """
        self._check_availability()
        
        # Check all input files exist
        for audio_path in audio_paths:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if output_audio_path is None:
            # Generate a filename
            output_filename = f"combined_audio_{len(audio_paths)}_files.wav"
            output_audio_path = os.path.join(self.cache_dir, output_filename)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        
        # Call the API to combine audio files
        try:
            result_path = self.api.combine_audio_files(
                input_paths=audio_paths,
                output_path=output_audio_path,
                instructions=instructions
            )
            
            logger.success(f"Successfully combined audio files: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Error combining audio files: {str(e)}")
            raise RuntimeError(f"Failed to combine audio files with WavCraft: {str(e)}")

    def create_audio_script(
        self, 
        script_prompt: str, 
        output_audio_path: Optional[str] = None,
        output_script_path: Optional[str] = None
    ) -> Dict[str, str]:
        """Create an audio script and the corresponding audio.
        
        Args:
            script_prompt (str): Prompt for the script to generate.
            output_audio_path (str, optional): Path to save the output audio.
                If None, a path will be generated.
            output_script_path (str, optional): Path to save the text script.
                If None, a path will be generated.
                
        Returns:
            Dict[str, str]: Dictionary with paths to the audio and script files.
        """
        self._check_availability()
        
        if output_audio_path is None:
            # Generate a filename based on the prompt
            safe_prompt = "".join(
                c if c.isalnum() else "_" for c in script_prompt[:20]
            )
            output_filename = f"script_{safe_prompt}.wav"
            output_audio_path = os.path.join(self.cache_dir, output_filename)
        
        if output_script_path is None:
            # Generate a script filename
            output_script_path = os.path.splitext(output_audio_path)[0] + ".txt"
        
        # Ensure the output directories exist
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_script_path), exist_ok=True)
        
        # Call the API to create audio script
        try:
            result = self.api.create_audio_script(
                prompt=script_prompt,
                output_audio_path=output_audio_path,
                output_script_path=output_script_path
            )
            
            logger.success(f"Successfully created audio script: {result['script_path']}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating audio script: {str(e)}")
            raise RuntimeError(f"Failed to create audio script with WavCraft: {str(e)}")

    def check_watermark(self, audio_path: str) -> Dict[str, Any]:
        """Check if an audio file contains a WavCraft watermark.
        
        Args:
            audio_path (str): Path to the audio file to check.
                
        Returns:
            Dict[str, Any]: Information about the watermark.
        """
        self._check_availability()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Call the API to check watermark
        try:
            return self.api.check_watermark(audio_path=audio_path)
            
        except Exception as e:
            logger.error(f"Error checking watermark: {str(e)}")
            return {"has_watermark": False, "error": str(e)}
    
    def _check_availability(self) -> None:
        """Check if the WavCraft API is available.
        
        Raises:
            RuntimeError: If the WavCraft API is not available
        """
        if not self.is_available:
            raise RuntimeError(
                "WavCraft API is not available. Please check the installation "
                "and provided path."
            )

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions
        in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the
                functions in the toolkit.
        """
        return [
            FunctionTool(self.edit_audio),
            FunctionTool(self.generate_audio),
            FunctionTool(self.combine_audio_files),
            FunctionTool(self.create_audio_script),
            FunctionTool(self.check_watermark),
        ] 