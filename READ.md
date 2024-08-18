# Tsuki AI Assistant

## Overview

Tsuki is an AI assistant application built with Python and Kivy. It leverages face recognition, speech recognition, and OpenAI's GPT-3.5-turbo to provide interactive responses and support. The application uses voice synthesis to communicate with users and can recognize faces to tailor responses based on user identity.

## Features

- **Face Recognition**: Recognizes and remembers users' faces.
- **Speech Recognition**: Listens to and processes user commands and queries.
- **Voice Synthesis**: Speaks responses to users with configurable voice properties.
- **OpenAI Integration**: Fetches and provides answers from OpenAI's GPT-3.5-turbo.
- **Image Processing**: Captures and processes video frames for face recognition.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- OpenCV
- `face_recognition`
- `pyttsx3`
- `speech_recognition`
- `aiohttp`
- `kivy`
- `openai`

You can install the required Python packages using:

```bash
pip install opencv-python face_recognition pyttsx3 SpeechRecognition aiohttp kivy openai
