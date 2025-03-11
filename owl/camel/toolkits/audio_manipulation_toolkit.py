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
import time
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import importlib.util
import torch
import random
import string
import soundfile as sf
import requests
import base64
from urllib.parse import urlparse

from loguru import logger

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool

# Third-party imports - these will be done lazily to avoid forcing users
# to install all dependencies if they don't use this toolkit
try:
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
    from transformers import pipeline
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False


class AudioLDMWrapper:
    """A wrapper for AudioLDM to generate audio from text."""
    
    def __init__(self, model_id: str = "cvssp/audioldm-s-full-v2"):
        """Initialize the AudioLDM wrapper.
        
        Args:
            model_id (str, optional): The AudioLDM model ID to use.
                Defaults to "cvssp/audioldm-s-full-v2".
        """
        try:
            from diffusers import AudioLDMPipeline
            self.pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            # Store the sample rate (usually 16000 Hz for AudioLDM models)
            self.sample_rate = 16000  # Default sample rate for AudioLDM
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
            self.is_available = True
        except Exception as e:
            logger.error(f"Failed to initialize AudioLDM: {e}")
            self.is_available = False
    
    def generate(
        self, 
        prompt: str, 
        output_path: str, 
        duration: int = 5,
        num_inference_steps: int = 25,
        guidance_scale: float = 2.5
    ) -> str:
        """Generate audio from a text prompt.
        
        Args:
            prompt (str): Text prompt describing the audio to generate
            output_path (str): Path to save the generated audio
            duration (int, optional): Duration of the audio in seconds.
                Defaults to 5.
            num_inference_steps (int, optional): Number of inference steps.
                Defaults to 25.
            guidance_scale (float, optional): Guidance scale. Defaults to 2.5.
            
        Returns:
            str: Path to the generated audio file
        """
        if not self.is_available:
            raise RuntimeError("AudioLDM is not available")
        
        # Generate audio
        audio = self.pipe(
            prompt=prompt, 
            num_inference_steps=num_inference_steps,
            audio_length_in_s=duration,
            guidance_scale=guidance_scale
        ).audios[0]
        
        # Save audio file using the stored sample rate instead of trying to access it from config
        sf.write(output_path, audio, self.sample_rate)
        
        return output_path


class AudioProcessing:
    """Audio processing utilities for manipulating audio files."""
    
    @staticmethod
    def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
        """Load an audio file.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        return librosa.load(file_path, sr=None)
    
    @staticmethod
    def save_audio(audio_data: np.ndarray, sample_rate: int, file_path: str) -> str:
        """Save audio data to a file.
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            file_path (str): Path to save the audio file
            
        Returns:
            str: Path to the saved audio file
        """
        sf.write(file_path, audio_data, sample_rate)
        return file_path
    
    @staticmethod
    def overlay_audio(
        background_path: str, 
        foreground_path: str, 
        output_path: str,
        position: float = 0.0,
        gain_adjustment: float = 0.0
    ) -> str:
        """Overlay one audio file on top of another.
        
        Args:
            background_path (str): Path to the background audio file
            foreground_path (str): Path to the foreground audio file
            output_path (str): Path to save the combined audio file
            position (float, optional): Position in seconds to add foreground.
                Defaults to 0.0.
            gain_adjustment (float, optional): Gain adjustment in dB.
                Defaults to 0.0.
                
        Returns:
            str: Path to the combined audio file
        """
        # Load audio files with pydub
        background = AudioSegment.from_file(background_path)
        foreground = AudioSegment.from_file(foreground_path)
        
        # Adjust gain if specified
        if gain_adjustment != 0.0:
            foreground = foreground.apply_gain(gain_adjustment)
        
        # Convert position to milliseconds
        position_ms = int(position * 1000)
        
        # Overlay foreground on background
        result = background.overlay(foreground, position=position_ms)
        
        # Export the result
        result.export(output_path, format="wav")
        
        return output_path
    
    @staticmethod
    def combine_audio_files(
        audio_paths: List[str], 
        output_path: str, 
        crossfade: int = 100
    ) -> str:
        """Combine multiple audio files.
        
        Args:
            audio_paths (List[str]): List of paths to audio files
            output_path (str): Path to save the combined audio file
            crossfade (int, optional): Crossfade in milliseconds.
                Defaults to 100.
                
        Returns:
            str: Path to the combined audio file
        """
        if not audio_paths:
            raise ValueError("No audio files provided")
        
        # Load the first audio file
        combined = AudioSegment.from_file(audio_paths[0])
        
        # Add remaining files with crossfade
        for audio_path in audio_paths[1:]:
            next_segment = AudioSegment.from_file(audio_path)
            combined = combined.append(next_segment, crossfade=crossfade)
        
        # Export the combined audio
        combined.export(output_path, format="wav")
        
        return output_path
    
    @staticmethod
    def apply_effect(
        input_path: str,
        output_path: str,
        effect_type: str,
        **effect_params
    ) -> str:
        """Apply an effect to an audio file.
        
        Args:
            input_path (str): Path to the input audio file
            output_path (str): Path to save the output audio file
            effect_type (str): Type of effect to apply
            **effect_params: Effect-specific parameters
            
        Returns:
            str: Path to the processed audio file
        """
        # Load the audio file
        audio = AudioSegment.from_file(input_path)
        
        # Apply the specified effect
        if effect_type == "fade_in":
            duration_ms = effect_params.get("duration_ms", 1000)
            audio = audio.fade_in(duration_ms)
        
        elif effect_type == "fade_out":
            duration_ms = effect_params.get("duration_ms", 1000)
            audio = audio.fade_out(duration_ms)
        
        elif effect_type == "speed":
            factor = effect_params.get("factor", 1.5)
            # This is a simple speed change without pitch correction
            # For better quality, a more sophisticated approach would be needed
            samples = np.array(audio.get_array_of_samples())
            audio = audio._spawn(samples[::int(1/factor)])
        
        elif effect_type == "volume":
            gain_db = effect_params.get("gain_db", 3.0)
            audio = audio.apply_gain(gain_db)
        
        elif effect_type == "reverb":
            # Simplified reverb effect - for production use, consider a DSP library
            delay_ms = effect_params.get("delay_ms", 100)
            decay = effect_params.get("decay", 0.5)
            
            delayed = audio[delay_ms:].apply_gain(-decay)
            audio = audio.overlay(delayed)
        
        else:
            raise ValueError(f"Unknown effect type: {effect_type}")
        
        # Export the processed audio
        audio.export(output_path, format="wav")
        
        return output_path


class TextToSpeechWrapper:
    """A wrapper for text-to-speech synthesis."""
    
    def __init__(self, model_id: str = "facebook/fastspeech2-en-ljspeech"):
        """Initialize the TTS wrapper.
        
        Args:
            model_id (str, optional): The TTS model ID to use.
                Defaults to "facebook/fastspeech2-en-ljspeech".
        """
        try:
            self.tts = pipeline("text-to-speech", model=model_id)
            self.is_available = True
        except Exception as e:
            logger.error(f"Failed to initialize text-to-speech: {e}")
            self.is_available = False
    
    def synthesize(self, text: str, output_path: str) -> str:
        """Synthesize speech from text.
        
        Args:
            text (str): Text to synthesize
            output_path (str): Path to save the generated audio
            
        Returns:
            str: Path to the generated audio file
        """
        if not self.is_available:
            raise RuntimeError("Text-to-speech is not available")
        
        # Generate speech
        speech = self.tts(text)
        
        # Save audio file
        with open(output_path, "wb") as f:
            f.write(speech["bytes"])
        
        return output_path


class AudioWatermark:
    """Audio watermarking utility."""
    
    @staticmethod
    def add_watermark(input_path: str, output_path: str) -> str:
        """Add a watermark to an audio file.
        
        Args:
            input_path (str): Path to the input audio file
            output_path (str): Path to save the watermarked audio
            
        Returns:
            str: Path to the watermarked audio file
        """
        # Load the audio file
        audio_data, sample_rate = librosa.load(input_path, sr=None)
        
        # Simple watermark by adding a very quiet high-frequency tone
        # This is a simplified example - production watermarking would
        # use more sophisticated techniques
        watermark_freq = 18000  # Hz (near upper limit of human hearing)
        duration = len(audio_data) / sample_rate
        t = np.arange(0, duration, 1/sample_rate)
        watermark = 0.001 * np.sin(2 * np.pi * watermark_freq * t)  # Very quiet
        
        # Ensure the watermark is the same length as the audio
        watermark = watermark[:len(audio_data)]
        
        # Add the watermark
        watermarked_audio = audio_data + watermark
        
        # Save the watermarked audio
        sf.write(output_path, watermarked_audio, sample_rate)
        
        return output_path
    
    @staticmethod
    def detect_watermark(audio_path: str) -> Dict[str, Any]:
        """Detect a watermark in an audio file.
        
        Args:
            audio_path (str): Path to the audio file to check
            
        Returns:
            Dict[str, Any]: Information about the watermark
        """
        # Load the audio file
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        
        # Analyze high frequencies where the watermark would be
        # This is a simplified detection - production watermark detection
        # would use more sophisticated techniques
        S = np.abs(librosa.stft(audio_data))
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sample_rate)
        
        # Look for energy around 18kHz (our watermark frequency)
        watermark_freq = 18000
        watermark_idx = np.argmin(np.abs(freqs - watermark_freq))
        
        # Calculate energy in the watermark band
        watermark_energy = np.mean(S[watermark_idx-5:watermark_idx+5, :])
        
        # Calculate energy in nearby bands for comparison
        nearby_energy = np.mean(S[watermark_idx-20:watermark_idx-10, :])
        
        # Watermark is detected if the energy in the watermark band
        # is significantly higher than nearby bands
        has_watermark = watermark_energy > (nearby_energy * 1.5)
        
        confidence = min(1.0, (watermark_energy / nearby_energy) - 1)
        
        return {
            "has_watermark": has_watermark,
            "confidence": confidence if has_watermark else 0.0,
            "message": f"Watermark {'detected' if has_watermark else 'not detected'}"
        }


class AudioManipulationToolkit(BaseToolkit):
    r"""A toolkit for audio manipulation using various audio libraries.
    
    This toolkit provides audio editing, generation, and manipulation capabilities
    using libraries like AudioLDM, librosa, and pydub instead of depending on
    external tools like WavCraft.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        audio_ldm_model: str = "cvssp/audioldm-s-full-v2",
        tts_model: str = "facebook/fastspeech2-en-ljspeech",
    ):
        """Initialize the audio manipulation toolkit.
        
        Args:
            cache_dir (str, optional): Directory to cache audio files.
                Defaults to 'tmp/'.
            audio_ldm_model (str, optional): AudioLDM model ID to use.
                Defaults to "cvssp/audioldm-s-full-v2".
            tts_model (str, optional): Text-to-speech model ID to use.
                Defaults to "facebook/fastspeech2-en-ljspeech".
        """
        if not HAS_AUDIO_LIBS:
            logger.warning(
                "Required audio libraries are not installed. "
                "Please install librosa, soundfile, pydub, and transformers."
            )
            self.is_available = False
            return
        
        self.cache_dir = cache_dir or "tmp/"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.is_available = True
        
        # Initialize components lazily
        self._audio_ldm = None
        self._tts = None
        self._audio_ldm_model = audio_ldm_model
        self._tts_model = tts_model
    
    @property
    def audio_ldm(self) -> AudioLDMWrapper:
        """Lazy initialization of AudioLDM."""
        if self._audio_ldm is None:
            self._audio_ldm = AudioLDMWrapper(model_id=self._audio_ldm_model)
        return self._audio_ldm
    
    @property
    def tts(self) -> TextToSpeechWrapper:
        """Lazy initialization of text-to-speech."""
        if self._tts is None:
            self._tts = TextToSpeechWrapper(model_id=self._tts_model)
        return self._tts

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
            # Generate a filename based on the instruction
            input_filename = os.path.basename(input_audio_path)
            input_name, input_ext = os.path.splitext(input_filename)
            safe_instruction = "".join(
                c if c.isalnum() else "_" for c in instruction[:20]
            )
            output_filename = f"{input_name}_{safe_instruction}{input_ext}"
            output_audio_path = os.path.join(self.cache_dir, output_filename)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        
        # Parse the instruction to determine what kind of edit to perform
        instruction_lower = instruction.lower()
        
        # Generate a temporary file for any additional audio
        effect_audio_path = None
        
        try:
            # Handle different types of edits based on instruction keywords
            if "add" in instruction_lower or "overlay" in instruction_lower or "mix" in instruction_lower:
                # Extract what to add from instruction
                to_add = instruction_lower.split("add", 1)[1].strip() if "add" in instruction_lower else instruction_lower.split("overlay", 1)[1].strip()
                
                # Generate the audio to add
                effect_audio_path = os.path.join(
                    self.cache_dir, 
                    f"effect_{''.join(random.choices(string.ascii_lowercase, k=8))}.wav"
                )
                self.audio_ldm.generate(to_add, effect_audio_path, duration=5)
                
                # Overlay the effect audio onto the input audio
                AudioProcessing.overlay_audio(
                    input_audio_path, 
                    effect_audio_path, 
                    output_audio_path,
                    position=1.0,  # Start 1 second in
                    gain_adjustment=-3.0  # Slightly quieter than the original
                )
            
            elif "reverb" in instruction_lower or "echo" in instruction_lower:
                # Apply reverb effect
                AudioProcessing.apply_effect(
                    input_audio_path,
                    output_audio_path,
                    "reverb",
                    delay_ms=100,
                    decay=0.5
                )
            
            elif "fade in" in instruction_lower:
                # Apply fade in effect
                AudioProcessing.apply_effect(
                    input_audio_path,
                    output_audio_path,
                    "fade_in",
                    duration_ms=1000  # 1-second fade
                )
            
            elif "fade out" in instruction_lower:
                # Apply fade out effect
                AudioProcessing.apply_effect(
                    input_audio_path,
                    output_audio_path,
                    "fade_out",
                    duration_ms=1000  # 1-second fade
                )
            
            elif "volume" in instruction_lower or "louder" in instruction_lower or "quieter" in instruction_lower:
                # Determine gain adjustment
                gain_db = 3.0  # Default to increasing volume
                if "quieter" in instruction_lower or "lower" in instruction_lower:
                    gain_db = -3.0
                
                # Apply volume adjustment
                AudioProcessing.apply_effect(
                    input_audio_path,
                    output_audio_path,
                    "volume",
                    gain_db=gain_db
                )
            
            elif "speed" in instruction_lower or "faster" in instruction_lower or "slower" in instruction_lower:
                # Determine speed factor
                factor = 1.5  # Default to increasing speed
                if "slower" in instruction_lower:
                    factor = 0.75
                
                # Apply speed adjustment
                AudioProcessing.apply_effect(
                    input_audio_path,
                    output_audio_path,
                    "speed",
                    factor=factor
                )
            
            else:
                # Default case - just add the audio watermark to the file
                # In a real implementation, we would use a more sophisticated
                # NLP approach to parse the instruction
                AudioWatermark.add_watermark(input_audio_path, output_audio_path)
                logger.warning(f"Couldn't parse instruction '{instruction}'. Applied only watermark.")
            
            # Add watermark to the result
            temp_output = output_audio_path + ".tmp"
            shutil.move(output_audio_path, temp_output)
            AudioWatermark.add_watermark(temp_output, output_audio_path)
            os.remove(temp_output)
            
            logger.success(f"Successfully edited audio: {output_audio_path}")
            return output_audio_path
            
        except Exception as e:
            logger.error(f"Error editing audio: {str(e)}")
            raise RuntimeError(f"Failed to edit audio: {str(e)}")
        
        finally:
            # Clean up temporary files
            if effect_audio_path and os.path.exists(effect_audio_path):
                os.remove(effect_audio_path)

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
                Defaults to 5 seconds if None.
                
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
        
        # Determine if this is speech or environmental audio
        is_speech = any(keyword in instruction.lower() for keyword in [
            "speak", "say", "voice", "speech", "narration", "talking",
            "conversation", "dialogue", "monologue"
        ])
        
        try:
            if is_speech:
                # Extract the text to speak
                # This is a simplified approach - in a real implementation,
                # you'd want to use a more sophisticated NLP approach
                text_to_speak = instruction
                for prefix in ["speak", "say", "voice", "speech", "narration"]:
                    if prefix in instruction.lower():
                        parts = instruction.lower().split(prefix, 1)
                        if len(parts) > 1:
                            text_to_speak = parts[1].strip()
                            break
                
                # Use text-to-speech for speech generation
                self.tts.synthesize(text_to_speak, output_audio_path)
            else:
                # Use AudioLDM for environmental sounds
                actual_duration = duration or 5  # Default to 5 seconds
                self.audio_ldm.generate(
                    instruction, 
                    output_audio_path, 
                    duration=int(actual_duration)
                )
            
            # Add watermark to the result
            temp_output = output_audio_path + ".tmp"
            shutil.move(output_audio_path, temp_output)
            AudioWatermark.add_watermark(temp_output, output_audio_path)
            os.remove(temp_output)
            
            logger.success(f"Successfully generated audio: {output_audio_path}")
            return output_audio_path
            
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise RuntimeError(f"Failed to generate audio: {str(e)}")

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
        
        try:
            # Determine crossfade duration if instructions provided
            crossfade = 100  # Default 100ms crossfade
            
            if instructions:
                instructions_lower = instructions.lower()
                if "crossfade" in instructions_lower:
                    # Extract crossfade value if provided
                    for word in instructions_lower.split():
                        if word.isdigit():
                            crossfade = int(word)
                            break
            
            # Combine the audio files
            AudioProcessing.combine_audio_files(
                audio_paths,
                output_audio_path,
                crossfade=crossfade
            )
            
            # Add watermark to the result
            temp_output = output_audio_path + ".tmp"
            shutil.move(output_audio_path, temp_output)
            AudioWatermark.add_watermark(temp_output, output_audio_path)
            os.remove(temp_output)
            
            logger.success(f"Successfully combined audio files: {output_audio_path}")
            return output_audio_path
            
        except Exception as e:
            logger.error(f"Error combining audio files: {str(e)}")
            raise RuntimeError(f"Failed to combine audio files: {str(e)}")


    def evaluate_audio(self, audio_path: str, criteria: str = None) -> str:
        r"""Evaluate the  audio file using GPT-4o-audio-preview.

        Args:
            audio_path (str): The path to the audio file.
            criteria (str, optional): Specific criteria to assess, e.g., 
                "clarity", "background noise", "pronunciation". Defaults to None
                which does a comprehensive quality assessment.

        Returns:
            str: A detailed assessment of the audio quality.
        """
        logger.debug(
            f"Evaluating audio quality for file `{audio_path}` with criteria: `{criteria}`"
        )

        parsed_url = urlparse(audio_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])
        encoded_string = None

        if is_url:
            res = requests.get(audio_path)
            res.raise_for_status()
            audio_data = res.content
            encoded_string = base64.b64encode(audio_data).decode('utf-8')
        else:
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            audio_file.close()
            encoded_string = base64.b64encode(audio_data).decode('utf-8')

        file_suffix = os.path.splitext(audio_path)[1]
        file_format = file_suffix[1:]

        # Prepare prompt based on criteria
        if criteria:
            text_prompt = f"""Evaluate the quality of this audio focusing specifically on {criteria}. 
            Provide a detailed assessment of strengths and weaknesses, with a numerical rating from 1-10 for each aspect.
            Include specific timestamps for any issues identified."""
        else:
            text_prompt = """Evaluate the quality of this audio across the following dimensions:
            1. Overall Clarity and Intelligibility (1-10)
            2. Background Noise Level (1-10, where 10 means no background noise)
            3. Voice/Sound Quality (1-10) 
            4. Pronunciation and Articulation if speech is present (1-10)
            5. Audio Balance and Levels (1-10)
            
            For each aspect, provide:
            - A numerical rating
            - Specific observations with timestamps where relevant
            - Suggestions for improvement
            
            End with an overall quality score and summary assessment."""

        completion = self.client.chat.completions.create(
            model="gpt-4o-audio-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an audio quality assessment expert. Provide detailed, technical, and actionable feedback on audio quality.",
                },
                {  # type: ignore[list-item, misc]
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_string,
                                "format": file_format,
                            },
                        },
                    ],
                },
            ],
        )  # type: ignore[misc]

        response: str = str(completion.choices[0].message.content)
        logger.debug(f"Audio quality assessment: {response}")
        return str(response)

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
        
        try:
            # Use text generation to create a script
            from transformers import pipeline
            
            # Initialize text generation model
            text_generator = pipeline("text-generation", model="gpt2-medium")
            
            # Generate the script text
            generated_text = text_generator(
                script_prompt,
                max_length=200,
                num_return_sequences=1
            )[0]["generated_text"]
            
            # Save the script to a file
            with open(output_script_path, "w", encoding="utf-8") as f:
                f.write(generated_text)
            
            # Generate audio from the script using text-to-speech
            self.tts.synthesize(generated_text, output_audio_path)
            
            # Add watermark to the audio
            temp_output = output_audio_path + ".tmp"
            shutil.move(output_audio_path, temp_output)
            AudioWatermark.add_watermark(temp_output, output_audio_path)
            os.remove(temp_output)
            
            logger.success(f"Successfully created audio script: {output_script_path}")
            return {
                "audio_path": output_audio_path,
                "script_path": output_script_path
            }
            
        except Exception as e:
            logger.error(f"Error creating audio script: {str(e)}")
            raise RuntimeError(f"Failed to create audio script: {str(e)}")

    def check_watermark(self, audio_path: str) -> Dict[str, Any]:
        """Check if an audio file contains a watermark.
        
        Args:
            audio_path (str): Path to the audio file to check.
                
        Returns:
            Dict[str, Any]: Information about the watermark.
        """
        self._check_availability()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Check for watermark
            return AudioWatermark.detect_watermark(audio_path)
            
        except Exception as e:
            logger.error(f"Error checking watermark: {str(e)}")
            return {"has_watermark": False, "error": str(e)}
    
    def _check_availability(self) -> None:
        """Check if the required audio libraries are available.
        
        Raises:
            RuntimeError: If the audio libraries are not available
        """
        if not self.is_available:
            raise RuntimeError(
                "Required audio libraries are not available. "
                "Please install librosa, soundfile, pydub, and transformers."
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
            FunctionTool(self.evaluate_audio)
        ] 