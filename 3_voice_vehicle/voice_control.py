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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz

nltk.download('punkt_tab')
nltk.download('punkt')

# Proper serial import
try:
    import serial.tools.list_ports
    import serial
except ImportError:
    print("PySerial not installed. Installing required packages...")
    import pip
    pip.main(['install', 'pyserial'])
    import serial.tools.list_ports
    import serial

class EnhancedVoiceControl:
    def __init__(self, port=None, baud_rate=9600):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Attempt to automatically find Arduino port
        self.arduino = None
        self.connect_to_arduino(port, baud_rate)
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
        
        # Initialize parameters
        self.wake_word = "vehicle"
        self.command_timeout = 5
        self.listening_active = False
        self.command_history = deque(maxlen=10)
        self.command_queue = Queue()
        self.confidence_threshold = 0.6
        
        # Enhanced command mapping with variations
        self.command_patterns = {
            'forward': {
                'variations': ['go forward', 'move forward', 'straight ahead', 'advance'],
                'action': 'F',
                'parameters': ['speed', 'duration']
            },
            'backward': {
                'variations': ['go back', 'move backward', 'reverse', 'retreat'],
                'action': 'B',
                'parameters': ['speed', 'duration']
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
        }
        
        # Start background processing if Arduino is connected
        if self.arduino:
            self.background_thread = threading.Thread(target=self._process_command_queue, daemon=True)
            self.background_thread.start()

    def connect_to_arduino(self, port=None, baud_rate=9600):
        """Attempt to connect to Arduino with error handling"""
        try:
            # If no port specified, try to find Arduino automatically
            if port is None:
                ports = list(serial.tools.list_ports.comports())
                for p in ports:
                    # Look for common Arduino identifiers
                    if "Arduino" in p.description or "CH340" in p.description or "USB Serial" in p.description:
                        port = p.device
                        break
                
                if port is None and ports:
                    # If no Arduino found but ports exist, use the first one
                    port = ports[0].device
                
                if port is None:
                    raise serial.SerialException("No serial ports found")

            print(f"Attempting to connect to {port}")
            self.arduino = serial.Serial(port=port, baudrate=baud_rate, timeout=1)
            time.sleep(2)  # Wait for connection to stabilize
            print(f"Successfully connected to {port}")
            
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            print("Running in test mode without Arduino connection")
            self.arduino = None
            
        except Exception as e:
            print(f"Unexpected error while connecting to Arduino: {e}")
            print("Running in test mode without Arduino connection")
            self.arduino = None

    def _process_command_queue(self):
        """Background thread to process command queue"""
        while True:
            if not self.command_queue.empty():
                command, params = self.command_queue.get()
                self._execute_command(command, params)
            time.sleep(0.1)
    
    def process_command_text(self, command_text):
        """Process recognized text to identify commands and parameters"""
        best_command = None
        best_confidence = 0
        command_params = {}
        
        # Clean and tokenize the command
        cleaned_text = command_text.lower().translate(
            str.maketrans('', '', string.punctuation)
        )
        
        # Check each command pattern
        for command, pattern in self.command_patterns.items():
            confidence = self.calculate_command_confidence(cleaned_text, pattern)
            
            if confidence > best_confidence and confidence > self.confidence_threshold:
                best_confidence = confidence
                best_command = pattern['action']
                command_params = self.extract_parameters(cleaned_text)
        
        if best_command:
            self.command_history.append((command_text, best_command, command_params))
            return best_command, command_params
        return None
    
    def _execute_command(self, command, params):
        """Execute command with parameters"""
        if not command:
            return
            
        # Format command with parameters
        command_str = command
        if 'speed' in params:
            command_str += f",{params['speed']}"
        else:
            command_str += ",200"  # default speed
            
        if 'duration' in params:
            command_str += f",{params['duration']}"
        if 'angle' in params:
            command_str += f",{params['angle']}"
        
        # Only try to send if Arduino is connected
        if self.arduino:
            try:
                self.arduino.write(command_str.encode())
            except Exception as e:
                print(f"Error sending command to Arduino: {e}")
        else:
            print(f"Test mode - Command would be: {command_str}")

    def run(self):
        """Main run loop with wake word detection"""
        print("Enhanced Voice Control System Started")
        while True:
            # Wait for wake word
            if self.listen_for_wake_word():
                print("Wake word detected!")
                # Listen for command
                command_result = self.listen_for_command()
                if command_result:
                    command, params = command_result
                    self.command_queue.put((command, params))
                
    def get_command_history(self):
        """Return recent command history"""
        return list(self.command_history)

    def close(self):
        """Cleanup resources"""
        try:
            if self.arduino:
                self.arduino.close()
        except Exception as e:
            print(f"Error closing connection: {e}")
        
    def listen_for_wake_word(self):
        """Listen specifically for wake word"""
        with sr.Microphone() as source:
            print("Listening for wake word...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            
        try:
            text = self.recognizer.recognize_google(audio).lower()
            return self.wake_word in text
        except:
            return False

    def listen_for_command(self):
        """Enhanced command listening with noise filtering"""
        with sr.Microphone() as source:
            print("Listening for command...")
            # Dynamic noise adjustment
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_adjustment_ratio = 1.5
            
            try:
                audio = self.recognizer.listen(source, timeout=self.command_timeout)
                text = self.recognizer.recognize_google(audio)
                return self.process_command_text(text)
            except sr.WaitTimeoutError:
                print("Listening timed out")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError:
                print("Could not request results")
                return None    
    
    def calculate_command_confidence(self, recognized_text, command_pattern):
        """Calculate confidence score for command matching"""
        max_ratio = 0
        for variation in command_pattern['variations']:
            ratio = fuzz.ratio(recognized_text.lower(), variation)
            max_ratio = max(max_ratio, ratio)
        return max_ratio / 100.0

    def extract_parameters(self, command_text):
        """Extract speed, duration, or angle parameters from command"""
        tokens = word_tokenize(command_text.lower())
        params = {}
        
        # Extract speed
        speed_words = ['fast', 'slow', 'medium', 'quickly', 'slowly']
        speed_map = {'fast': 255, 'quickly': 255, 'medium': 150, 'slow': 100, 'slowly': 100}
        
        for word in tokens:
            if word in speed_words:
                params['speed'] = speed_map[word]
                
        # Extract duration
        for i, token in enumerate(tokens):
            if token.isdigit() and i+1 < len(tokens):
                if tokens[i+1] in ['seconds', 'second', 'sec']:
                    params['duration'] = int(token)
                elif tokens[i+1] in ['degrees', 'degree']:
                    params['angle'] = int(token)
                    
        return params
    
if __name__ == "__main__":
    controller = EnhancedVoiceControl()
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.close()
        print("\nSystem stopped")