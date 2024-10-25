import speech_recognition as sr
import serial
import time
import threading
from collections import deque
import numpy as np
from queue import Queue
import sounddevice as sd
import string
import nltk
import json
import yaml
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_control.log'),
        logging.StreamHandler()
    ]
)

class Config:
    """Configuration management class"""
    DEFAULT_CONFIG = {
        'wake_word': 'control',
        'command_timeout': 5,
        'confidence_threshold': 0.6,
        'default_speed': 200,
        'retry_attempts': 3,
        'retry_delay': 1,
        'baud_rate': 9600,
        'command_patterns': {
            'forward': {
                'variations': ['go forward', 'move forward', 'straight ahead', 'advance'],
                'action': 'F',
                'parameters': ['speed', 'duration', 'distance']
            },
            'backward': {
                'variations': ['go back', 'move backward', 'reverse', 'retreat'],
                'action': 'B',
                'parameters': ['speed', 'duration', 'distance']
            },
            'left': {
                'variations': ['turn left', 'go left', 'rotate left'],
                'action': 'L',
                'parameters': ['angle', 'speed']
            },
            'right': {
                'variations': ['turn right', 'go right', 'rotate right'],
                'action': 'R',
                'parameters': ['angle', 'speed']
            },
            'stop': {
                'variations': ['halt', 'freeze', 'stay', 'brake'],
                'action': 'S',
                'parameters': []
            }
        },
        'speed_mappings': {
            'fast': 255,
            'quickly': 255,
            'medium': 150,
            'slow': 100,
            'slowly': 100
        }
    }

    @classmethod
    def load(cls, config_path='config.yaml'):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return {**cls.DEFAULT_CONFIG, **config}
        except FileNotFoundError:
            logging.info("No config file found, using defaults")
            return cls.DEFAULT_CONFIG

class ArduinoCommunicator:
    """Handles all Arduino communication"""
    def __init__(self, config):
        self.config = config
        self.arduino = None
        self.connect()

    def connect(self):
        """Attempt to connect to Arduino with retry logic"""
        for attempt in range(self.config['retry_attempts']):
            try:
                ports = list(serial.tools.list_ports.comports())
                for p in ports:
                    if any(id in p.description for id in ["Arduino", "CH340", "USB Serial"]):
                        self.arduino = serial.Serial(
                            port=p.device,
                            baudrate=self.config['baud_rate'],
                            timeout=1
                        )
                        time.sleep(2)
                        logging.info(f"Connected to Arduino on {p.device}")
                        return True
                raise serial.SerialException("No Arduino found")
            except Exception as e:
                logging.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.config['retry_attempts'] - 1:
                    time.sleep(self.config['retry_delay'])
        logging.warning("Running in test mode without Arduino connection")
        return False

    def send_command(self, command_str):
        """Send command with acknowledgment check"""
        if not self.arduino:
            logging.info(f"Test mode - Command would be: {command_str}")
            return True

        try:
            self.arduino.write(command_str.encode())
            # Wait for acknowledgment
            response = self.arduino.readline().decode().strip()
            if response == "OK":
                logging.info(f"Command {command_str} executed successfully")
                return True
            logging.warning(f"Unexpected response: {response}")
            return False
        except Exception as e:
            logging.error(f"Error sending command: {e}")
            return False

    def close(self):
        """Safely close Arduino connection"""
        if self.arduino:
            try:
                self.arduino.close()
                logging.info("Arduino connection closed")
            except Exception as e:
                logging.error(f"Error closing Arduino connection: {e}")

class CommandProcessor:
    """Handles command processing and validation"""
    def __init__(self, config):
        self.config = config
        self.command_history = deque(maxlen=10)

    def process_command_text(self, command_text):
        """Process recognized text to identify commands and parameters"""
        best_command = None
        best_confidence = 0
        command_params = {}
        
        cleaned_text = command_text.lower().translate(
            str.maketrans('', '', string.punctuation)
        )
        
        # Handle compound commands
        commands = self.split_compound_commands(cleaned_text)
        if len(commands) > 1:
            return [self.process_single_command(cmd) for cmd in commands]
        
        return self.process_single_command(cleaned_text)

    def process_single_command(self, cleaned_text):
        """Process a single command"""
        best_command = None
        best_confidence = 0
        command_params = {}
        
        for command, pattern in self.config['command_patterns'].items():
            confidence = self.calculate_command_confidence(cleaned_text, pattern)
            
            if confidence > best_confidence and confidence > self.config['confidence_threshold']:
                best_confidence = confidence
                best_command = pattern['action']
                command_params = self.extract_parameters(cleaned_text)
        
        if best_command:
            self.command_history.append((cleaned_text, best_command, command_params))
            return best_command, command_params
        return None

    def split_compound_commands(self, text):
        """Split text into multiple commands if compound command detected"""
        separators = ['then', 'after that', 'followed by', 'and then']
        commands = [text]
        for sep in separators:
            new_commands = []
            for cmd in commands:
                new_commands.extend(cmd.split(sep))
            commands = new_commands
        return [cmd.strip() for cmd in commands if cmd.strip()]

    def calculate_command_confidence(self, recognized_text, command_pattern):
        """Calculate confidence score for command matching"""
        max_ratio = 0
        for variation in command_pattern['variations']:
            ratio = fuzz.ratio(recognized_text.lower(), variation)
            max_ratio = max(max_ratio, ratio)
        return max_ratio / 100.0

    def extract_parameters(self, command_text):
        """Extract parameters from command text"""
        tokens = word_tokenize(command_text.lower())
        params = {}
        
        # Extract speed
        for word in tokens:
            if word in self.config['speed_mappings']:
                params['speed'] = self.config['speed_mappings'][word]
                
        # Extract numeric parameters
        for i, token in enumerate(tokens):
            if token.isdigit() and i+1 < len(tokens):
                next_word = tokens[i+1]
                if next_word in ['seconds', 'second', 'sec']:
                    params['duration'] = int(token)
                elif next_word in ['degrees', 'degree']:
                    params['angle'] = int(token)
                elif next_word in ['meters', 'meter', 'm']:
                    params['distance'] = int(token)
                    
        return params

class EnhancedVoiceControl:
    """Main voice control class with enhanced features"""
    def __init__(self, config_path='config.yaml'):
        self.config = Config.load(config_path)
        self.recognizer = sr.Recognizer()
        self.arduino_comm = ArduinoCommunicator(self.config)
        self.command_processor = CommandProcessor(self.config)
        self.command_queue = Queue()
        self.status = {'running': True, 'listening': False}
        
        # Start background processing
        self.background_thread = threading.Thread(target=self._process_command_queue, daemon=True)
        self.background_thread.start()

    def _process_command_queue(self):
        """Background thread to process command queue"""
        while self.status['running']:
            if not self.command_queue.empty():
                command, params = self.command_queue.get()
                self._execute_command(command, params)
            time.sleep(0.1)

    def _execute_command(self, command, params):
        """Execute command with parameters and handle compound commands"""
        if isinstance(command, list):  # Compound command
            for cmd, prm in command:
                self._execute_single_command(cmd, prm)
        else:
            self._execute_single_command(command, params)

    def _execute_single_command(self, command, params):
        """Execute a single command with retry logic"""
        if not command:
            return

        command_str = self._format_command(command, params)
        
        for attempt in range(self.config['retry_attempts']):
            if self.arduino_comm.send_command(command_str):
                return
            logging.warning(f"Retry attempt {attempt + 1} for command {command_str}")
            time.sleep(self.config['retry_delay'])
        
        logging.error(f"Failed to execute command {command_str} after {self.config['retry_attempts']} attempts")

    def _format_command(self, command, params):
        """Format command string with parameters"""
        command_str = command
        command_str += f",{params.get('speed', self.config['default_speed'])}"
        
        if 'duration' in params:
            command_str += f",{params['duration']}"
        if 'angle' in params:
            command_str += f",{params['angle']}"
        if 'distance' in params:
            command_str += f",{params['distance']}"
            
        return command_str

    def run(self):
        """Main run loop with enhanced error handling"""
        logging.info("Enhanced Voice Control System Started")
        while self.status['running']:
            try:
                if self.listen_for_wake_word():
                    logging.info("Wake word detected!")
                    self.status['listening'] = True
                    command_result = self.listen_for_command()
                    if command_result:
                        self.command_queue.put(command_result)
                    self.status['listening'] = False
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(1)

    def listen_for_wake_word(self):
        """Listen for wake word with noise adaptation"""
        with sr.Microphone() as source:
            logging.info("Listening for wake word...")
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                text = self.recognizer.recognize_google(audio).lower()
                return self.config['wake_word'] in text
            except Exception as e:
                logging.debug(f"Wake word detection cycle: {e}")
                return False

    def listen_for_command(self):
        """Enhanced command listening with noise filtering"""
        with sr.Microphone() as source:
            logging.info("Listening for command...")
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_adjustment_ratio = 1.5
            
            try:
                audio = self.recognizer.listen(source, timeout=self.config['command_timeout'])
                text = self.recognizer.recognize_google(audio)
                return self.command_processor.process_command_text(text)
            except sr.WaitTimeoutError:
                logging.info("Listening timed out")
            except sr.UnknownValueError:
                logging.info("Could not understand audio")
            except sr.RequestError as e:
                logging.error(f"Could not request results: {e}")
            return None

    def get_command_history(self):
        """Return recent command history"""
        return list(self.command_processor.command_history)

    def close(self):
        """Cleanup resources"""
        self.status['running'] = False
        self.arduino_comm.close()
        logging.info("System stopped")

if __name__ == "__main__":
    try:
        controller = EnhancedVoiceControl()
        controller.run()
    except KeyboardInterrupt:
        controller.close()
        logging.info("\nSystem stopped by user")